import asyncio
import base64
import json
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Iterator, Optional

import boto3
import uvicorn
from conversation_store import ConversationMessage, ConversationStore
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from llm_client import GrokConfig, GrokLLMClient, TalosConfig, TalosLLMClient
from logic import is_addressed, strip_wake_words, wake_word_required
from pydantic import BaseModel, Field
from prompt_builder import (
    PromptParticipant,
    PromptSpeaker,
    build_model_input,
    build_prompt_context,
)
from sentence_chunker import SentenceChunker

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("python-service")


@dataclass
class Settings:
    host: str = os.getenv("PYTHON_HOST", "127.0.0.1")
    port: int = int(os.getenv("PYTHON_PORT", "8000"))
    whisper_model: str = os.getenv("FASTER_WHISPER_MODEL", "base.en")
    whisper_device: str = os.getenv("FASTER_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8")
    whisper_fallback_cpu: bool = os.getenv(
        "FASTER_WHISPER_FALLBACK_CPU", "1"
    ).strip().lower() not in ("0", "false", "no", "")
    aws_region: str = os.getenv(
        "AWS_REGION",
        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    polly_voice_id: str = os.getenv("POLLY_VOICE_ID", "Brian")
    polly_engine: str = os.getenv("POLLY_ENGINE", "neural")
    # TTS transport. "pcm" returns raw 16-bit signed LE mono PCM (no MP3
    # encode/decode round-trip); "mp3" restores the legacy container.
    tts_format: str = os.getenv("TTS_AUDIO_FORMAT", "pcm").strip().lower()
    # Polly only supports 8000/16000 for pcm, so 16000 is the ceiling here.
    tts_pcm_sample_rate: int = int(os.getenv("POLLY_PCM_SAMPLE_RATE", "16000"))
    xai_api_key: Optional[str] = os.getenv("XAI_API_KEY")
    xai_model: str = os.getenv("XAI_MODEL", "grok-4-1-fast-reasoning")
    xai_base_url: str = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    xai_timeout_seconds: float = float(
        os.getenv("XAI_TIMEOUT_SECONDS", "20")
    )
    xai_max_output_tokens: int = int(os.getenv("XAI_MAX_OUTPUT_TOKENS", "180"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "24"))
    # Reply backend: "talos" routes to the TALOS brain over HTTP; "grok" (or any
    # other value) uses the hosted OpenAI-Responses client above.
    llm_provider: str = os.getenv("LLM_PROVIDER", "talos").strip().lower()
    # TALOS text agent (POST /chat) — same endpoint the TALOS GUI/voice worker use.
    talos_base_url: str = os.getenv(
        "TALOS_TEXT_AGENT_URL", "http://127.0.0.1:8420"
    )
    talos_token: str = os.getenv(
        "TALOS_TEXT_AGENT_TOKEN", os.getenv("TEXT_AGENT_API_TOKEN", "")
    )
    talos_source: str = os.getenv("TALOS_SOURCE", "discord")
    talos_mode: str = os.getenv("TALOS_MODE", "auto")
    talos_timeout_seconds: float = float(
        os.getenv("TALOS_TIMEOUT_SECONDS", "30")
    )
    talos_stream_path: str = os.getenv("TALOS_STREAM_PATH", "/chat/stream")


settings = Settings()
app = FastAPI(title="Discord Voice Prototype Service")


class CallUser(BaseModel):
    discord_user_id: Optional[int] = None
    username: Optional[str] = None
    display_name: Optional[str] = None


class AudioProcessRequest(BaseModel):
    guild_id: int
    guild_name: Optional[str] = None
    voice_channel_id: Optional[int] = None
    voice_channel_name: Optional[str] = None
    speaker_id: str
    discord_user_id: Optional[int] = None
    discord_username: Optional[str] = None
    discord_display_name: Optional[str] = None
    users_in_call: list[CallUser] = Field(default_factory=list)
    ssrc: int
    speaker_resolution: str
    utterance_id: int
    sample_rate: int
    channels: int
    audio_base64: str


class AudioProcessResponse(BaseModel):
    transcript: str
    should_respond: bool
    ignore_reason: Optional[str] = None
    reply_text: Optional[str] = None
    tts_audio_base64: Optional[str] = None
    tts_audio_format: Optional[str] = None
    # For raw PCM these tell the Rust bot how to build the Songbird input.
    tts_sample_rate: Optional[int] = None
    tts_channels: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    whisper_model: str


def _whisper_candidates(config: Settings) -> list[tuple[str, str]]:
    """Ordered (device, compute_type) attempts, most-preferred first.

    On CUDA we try the requested compute type, then plain ``float16`` (some GPUs
    lack the int8 GEMM path and raise CUBLAS_STATUS_NOT_SUPPORTED at inference,
    not at load), then CPU/int8 if fallback is enabled. Duplicates are removed.
    """
    device = config.whisper_device
    compute_type = config.whisper_compute_type

    if not device.startswith("cuda"):
        return [(device, compute_type)]

    gpu_visible = False
    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
        gpu_visible = count > 0
        logger.info("CUDA device_count=%s", count)
    except Exception as error:  # pragma: no cover - depends on runtime install
        logger.warning("could not query CUDA devices via ctranslate2: %s", error)

    candidates: list[tuple[str, str]] = []
    if gpu_visible:
        _log_gpu_free_vram()
        candidates.append((device, compute_type))
        candidates.append((device, "float16"))
        candidates.append((device, "int8"))
    else:
        logger.warning("no CUDA device visible")

    if config.whisper_fallback_cpu or not gpu_visible:
        candidates.append(("cpu", "int8"))

    # De-duplicate while preserving order.
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique or [(device, compute_type)]


def _warmup_whisper(model: WhisperModel) -> None:
    """Force one real encode so an unsupported GPU compute type fails here (at
    startup) rather than on the first live utterance."""
    import numpy as np

    audio = (np.random.default_rng(0).standard_normal(16_000).astype("float32")) * 0.01
    segments, _info = model.transcribe(audio, language="en", vad_filter=False)
    # Draining the generator is what triggers the encode/cuBLAS call.
    list(segments)


def load_whisper_model(config: Settings) -> WhisperModel:
    """Load faster-whisper, verifying each candidate config with a warmup so we
    fall back gracefully instead of 500-ing on every request."""
    candidates = _whisper_candidates(config)
    last_error: Optional[Exception] = None
    for device, compute_type in candidates:
        try:
            logger.info(
                "loading faster-whisper model=%s device=%s compute_type=%s",
                config.whisper_model,
                device,
                compute_type,
            )
            model = WhisperModel(
                config.whisper_model,
                device=device,
                compute_type=compute_type,
            )
            _warmup_whisper(model)
            logger.info(
                "faster-whisper ready device=%s compute_type=%s", device, compute_type
            )
            return model
        except Exception as error:
            last_error = error
            logger.warning(
                "faster-whisper failed on device=%s compute_type=%s: %s",
                device,
                compute_type,
                error,
            )
    raise RuntimeError(
        f"could not initialize faster-whisper on any candidate config: {last_error}"
    )


def _log_gpu_free_vram() -> None:
    """Best-effort log of per-GPU free VRAM; silently skips if pynvml is absent."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                logger.info(
                    "gpu %s free_vram=%.0fMB total_vram=%.0fMB",
                    index,
                    mem.free / (1024 * 1024),
                    mem.total / (1024 * 1024),
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # pynvml is optional; ctranslate2's device count is the hard requirement.
        pass


class VoiceService:
    def __init__(self, config: Settings) -> None:
        self.config = config
        self.model = load_whisper_model(config)
        self.polly = boto3.client("polly", region_name=config.aws_region)
        logger.info(
            "configured aws polly region=%s voice=%s engine=%s tts_format=%s pcm_sample_rate=%s",
            config.aws_region,
            config.polly_voice_id,
            config.polly_engine,
            config.tts_format,
            config.tts_pcm_sample_rate,
        )
        self.provider = config.llm_provider
        if self.provider == "talos":
            self.llm = TalosLLMClient(
                TalosConfig(
                    base_url=config.talos_base_url,
                    token=config.talos_token,
                    source=config.talos_source,
                    mode=config.talos_mode,
                    timeout_seconds=config.talos_timeout_seconds,
                    stream_path=config.talos_stream_path,
                )
            )
            logger.info(
                "configured llm provider=talos base_url=%s source=%s mode=%s timeout_seconds=%s",
                config.talos_base_url,
                config.talos_source,
                config.talos_mode,
                config.talos_timeout_seconds,
            )
        else:
            self.llm = GrokLLMClient(
                GrokConfig(
                    api_key=config.xai_api_key,
                    model=config.xai_model,
                    base_url=config.xai_base_url,
                    timeout_seconds=config.xai_timeout_seconds,
                    max_output_tokens=config.xai_max_output_tokens,
                )
            )
            logger.info(
                "configured llm provider=grok enabled=%s model=%s base_url=%s timeout_seconds=%s",
                self.llm.enabled,
                config.xai_model,
                config.xai_base_url,
                config.xai_timeout_seconds,
            )
        self.conversations = ConversationStore(
            max_history_messages=config.max_history_messages,
        )
        logger.info(
            "conversation history max_history_messages=%s",
            config.max_history_messages,
        )

    def transcribe_wav(self, wav_path: Path) -> str:
        segments, info = self.model.transcribe(
            str(wav_path),
            vad_filter=True,
            language="en",
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        logger.info(
            "transcript generated language=%s duration=%.2fs text=%r",
            getattr(info, "language", "unknown"),
            getattr(info, "duration", 0.0),
            transcript,
        )
        return transcript

    def synthesize_audio_sync(self, text: str) -> tuple[bytes, str, int, int]:
        """Synthesize ``text`` with Polly, returning (bytes, format, sample_rate, channels).

        With ``tts_format=pcm`` we ask Polly for raw 16-bit signed LE mono PCM so
        the Rust bot can feed it straight to Songbird — no MP3 encode on Polly and
        no MP3 decode in Songbird.
        """
        if self.config.tts_format == "pcm":
            output_format = "pcm"
            sample_rate = self.config.tts_pcm_sample_rate
        else:
            output_format = "mp3"
            sample_rate = 24000
        response = self.polly.synthesize_speech(
            Engine=self.config.polly_engine,
            OutputFormat=output_format,
            SampleRate=str(sample_rate),
            Text=text,
            TextType="text",
            VoiceId=self.config.polly_voice_id,
        )
        audio_stream = response["AudioStream"]
        try:
            audio_bytes = audio_stream.read()
        finally:
            audio_stream.close()
        # Polly pcm output is always single-channel.
        channels = 1
        logger.info(
            "tts generated format=%s sample_rate=%s channels=%s bytes=%s",
            output_format,
            sample_rate,
            channels,
            len(audio_bytes),
        )
        return audio_bytes, output_format, sample_rate, channels

    async def synthesize_audio(self, text: str) -> tuple[bytes, str, int, int]:
        return await asyncio.to_thread(self.synthesize_audio_sync, text)

    def _prepare_llm_call(
        self,
        request: AudioProcessRequest,
        recent_messages: list[ConversationMessage],
        user_text: str,
        addressed: bool,
        now: datetime,
        streaming: bool,
    ) -> tuple[list[dict[str, object]], str, str]:
        """Build the (model_input, session_id, message) tuple for an LLM call.

        Shared by the blocking and streaming paths so both providers behave
        identically regardless of transport.
        """
        speaker_name = (
            request.discord_display_name
            or request.discord_username
            or request.speaker_id
        )
        session_id = f"discord-guild-{request.guild_id}"

        if self.provider == "talos":
            # TALOS keeps its own history/personality, so send just the utterance.
            # In multi-user calls, prefix the speaker so TALOS can attribute turns.
            message = user_text
            if len(request.users_in_call) > 1:
                message = f"{speaker_name}: {user_text}"
            logger.info(
                "calling talos guild_id=%s utterance_id=%s speaker=%s participants=%s addressed=%s session_id=%s streaming=%s",
                request.guild_id,
                request.utterance_id,
                speaker_name,
                len(request.users_in_call),
                addressed,
                session_id,
                streaming,
            )
            return [], session_id, message

        prompt_context = build_prompt_context(
            guild_id=request.guild_id,
            guild_name=request.guild_name,
            voice_channel_id=request.voice_channel_id,
            voice_channel_name=request.voice_channel_name,
            now=now,
            current_speaker=PromptSpeaker(
                speaker_id=request.speaker_id,
                discord_user_id=request.discord_user_id,
                username=request.discord_username,
                display_name=request.discord_display_name or speaker_name,
            ),
            participants=[
                PromptParticipant(
                    discord_user_id=user.discord_user_id,
                    username=user.username,
                    display_name=user.display_name,
                )
                for user in request.users_in_call
            ],
            recent_messages=recent_messages,
            latest_user_utterance=user_text,
            addressed=addressed,
        )
        model_input = build_model_input(prompt_context)
        logger.info(
            "calling llm guild_id=%s utterance_id=%s speaker=%s participants=%s history_messages=%s addressed=%s streaming=%s",
            request.guild_id,
            request.utterance_id,
            speaker_name,
            len(request.users_in_call),
            len(recent_messages),
            addressed,
            streaming,
        )
        return model_input, session_id, user_text

    def generate_reply(
        self,
        request: AudioProcessRequest,
        recent_messages: list[ConversationMessage],
        user_text: str,
        addressed: bool,
        now: datetime,
    ) -> str:
        model_input, session_id, message = self._prepare_llm_call(
            request, recent_messages, user_text, addressed, now, streaming=False
        )
        return self.llm.generate_reply(
            model_input, session_id=session_id, user_text=message
        )

    def stream_reply(
        self,
        request: AudioProcessRequest,
        recent_messages: list[ConversationMessage],
        user_text: str,
        addressed: bool,
        now: datetime,
    ) -> Iterator[str]:
        model_input, session_id, message = self._prepare_llm_call(
            request, recent_messages, user_text, addressed, now, streaming=True
        )
        return self.llm.stream_reply(
            model_input, session_id=session_id, user_text=message
        )


voice_service = VoiceService(settings)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    logger.info("health check")
    return HealthResponse(status="ok", whisper_model=settings.whisper_model)


@app.post("/process-audio", response_model=AudioProcessResponse)
async def process_audio(request: AudioProcessRequest) -> AudioProcessResponse:
    audio_bytes = base64.b64decode(request.audio_base64)
    speaker_label = (
        request.discord_display_name
        or request.discord_username
        or request.speaker_id
    )
    logger.info(
        "received audio chunk guild_id=%s guild_name=%s channel_id=%s channel_name=%s utterance_id=%s speaker_id=%s discord_user_id=%s display_name=%s ssrc=%s resolved_via=%s participants=%s bytes=%s sample_rate=%s channels=%s",
        request.guild_id,
        request.guild_name,
        request.voice_channel_id,
        request.voice_channel_name,
        request.utterance_id,
        request.speaker_id,
        request.discord_user_id,
        speaker_label,
        request.ssrc,
        request.speaker_resolution,
        len(request.users_in_call),
        len(audio_bytes),
        request.sample_rate,
        request.channels,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav_path = Path(temp_file.name)
        temp_file.write(audio_bytes)

    try:
        transcript = await asyncio.to_thread(voice_service.transcribe_wav, wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    if not transcript:
        logger.info(
            "ignoring utterance_id=%s speaker=%s reason=empty_transcript",
            request.utterance_id,
            speaker_label,
        )
        return AudioProcessResponse(
            transcript="",
            should_respond=False,
            ignore_reason="empty_transcript",
        )

    now = datetime.now().astimezone()
    participant_count = len(request.users_in_call)
    requires_wake_word = wake_word_required(participant_count)
    addressed = is_addressed(transcript) if requires_wake_word else True
    recent_messages = voice_service.conversations.recent_messages(request.guild_id)
    should_attempt_reply = addressed
    user_text = strip_wake_words(transcript) if addressed else transcript

    voice_service.conversations.append(
        request.guild_id,
        ConversationMessage(
            timestamp=now,
            speaker_id=request.speaker_id,
            speaker_name=speaker_label,
            role="user",
            text=user_text,
        ),
    )
    logger.info(
        "transcript analyzed utterance_id=%s speaker=%s transcript=%r participants=%s requires_wake_word=%s addressed=%s should_attempt_reply=%s",
        request.utterance_id,
        speaker_label,
        transcript,
        participant_count,
        requires_wake_word,
        addressed,
        should_attempt_reply,
    )

    if not should_attempt_reply:
        logger.info(
            "ignoring utterance_id=%s speaker=%s reason=not_addressed_and_no_recent_followup",
            request.utterance_id,
            speaker_label,
        )
        return AudioProcessResponse(
            transcript=transcript,
            should_respond=False,
            ignore_reason="not_addressed",
        )

    try:
        reply_text = await asyncio.to_thread(
            voice_service.generate_reply,
            request,
            recent_messages,
            user_text,
            addressed,
            now,
        )
    except Exception as error:
        logger.exception(
            "llm reply generation failed guild_id=%s utterance_id=%s speaker=%s",
            request.guild_id,
            request.utterance_id,
            speaker_label,
        )
        fallback_reply = (
            "I'm having trouble responding right now. Please try again in a moment."
        )
        try:
            tts_audio, tts_format, tts_sample_rate, tts_channels = (
                await voice_service.synthesize_audio(fallback_reply)
            )
        except Exception:
            logger.exception(
                "fallback tts synthesis failed guild_id=%s utterance_id=%s",
                request.guild_id,
                request.utterance_id,
            )
            return AudioProcessResponse(
                transcript=transcript,
                should_respond=False,
                ignore_reason=f"llm_error:{type(error).__name__}",
            )

        voice_service.conversations.append(
            request.guild_id,
            ConversationMessage(
                timestamp=now,
                speaker_id="assistant:butler",
                speaker_name="Butler",
                role="assistant",
                text=fallback_reply,
            ),
        )
        return AudioProcessResponse(
            transcript=transcript,
            should_respond=True,
            reply_text=fallback_reply,
            tts_audio_base64=base64.b64encode(tts_audio).decode("utf-8"),
            tts_audio_format=tts_format,
            tts_sample_rate=tts_sample_rate,
            tts_channels=tts_channels,
        )

    try:
        tts_audio, tts_format, tts_sample_rate, tts_channels = (
            await voice_service.synthesize_audio(reply_text)
        )
    except Exception:
        logger.exception(
            "tts synthesis failed guild_id=%s utterance_id=%s",
            request.guild_id,
            request.utterance_id,
        )
        return AudioProcessResponse(
            transcript=transcript,
            should_respond=False,
            ignore_reason="tts_error",
            reply_text=reply_text,
        )
    voice_service.conversations.append(
        request.guild_id,
        ConversationMessage(
            timestamp=now,
            speaker_id="assistant:butler",
            speaker_name="Butler",
            role="assistant",
            text=reply_text,
        ),
    )
    return AudioProcessResponse(
        transcript=transcript,
        should_respond=True,
        reply_text=reply_text,
        tts_audio_base64=base64.b64encode(tts_audio).decode("utf-8"),
        tts_audio_format=tts_format,
        tts_sample_rate=tts_sample_rate,
        tts_channels=tts_channels,
    )


# --- Streaming endpoint ------------------------------------------------------
# Each line of the response body is a JSON object (NDJSON):
#   {"type": "meta",  "transcript": ..., "should_respond": bool, "ignore_reason": ...}
#   {"type": "audio", "seq": int, "text": ..., "audio_format": "pcm",
#                     "sample_rate": int, "channels": int, "pcm_base64": ...}
#   {"type": "end",   "reply_text": ..., "chunks": int}
#   {"type": "error", "message": ...}
# The Rust bot enqueues each "audio" frame into Songbird as it arrives, so
# playback begins on the first sentence while the rest is still generating.

_STREAM_SENTINEL = object()


def _ndjson(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def _synthesize_sentence_frame(sentence: str, seq: int) -> Optional[dict]:
    """Synthesize one sentence and package it as an audio frame.

    Returns ``None`` (and logs) on synth failure so one bad sentence doesn't tear
    down the whole stream — the remaining sentences still play.
    """
    sentence = sentence.strip()
    if not sentence:
        return None
    try:
        audio_bytes, fmt, sample_rate, channels = (
            voice_service.synthesize_audio_sync(sentence)
        )
    except Exception:
        logger.exception("streaming tts synthesis failed seq=%s text=%r", seq, sentence)
        return None
    return {
        "type": "audio",
        "seq": seq,
        "text": sentence,
        "audio_format": fmt,
        "sample_rate": sample_rate,
        "channels": channels,
        "pcm_base64": base64.b64encode(audio_bytes).decode("utf-8"),
    }


async def _reply_stream(
    request: AudioProcessRequest,
    recent_messages: list[ConversationMessage],
    user_text: str,
    addressed: bool,
    now: datetime,
    transcript: str,
    speaker_label: str,
    t_recv: float,
) -> AsyncIterator[bytes]:
    # Tell the bot up front that a reply is coming so it can prime playback.
    yield _ndjson(
        {
            "type": "meta",
            "transcript": transcript,
            "should_respond": True,
            "ignore_reason": None,
        }
    )

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(obj: object) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, obj)

    def worker() -> None:
        # Runs off the event loop: the Talos SSE read and each Polly synth are
        # blocking, and we want to overlap "generate next sentence" with "play
        # current sentence".
        full_parts: list[str] = []
        seq = 0
        # Latency milestones (perf_counter, seconds) relative to request receipt.
        t_first_token: Optional[float] = None
        t_first_audio: Optional[float] = None

        def handle_sentence(sentence: str, current_seq: int) -> int:
            nonlocal t_first_audio
            frame = _synthesize_sentence_frame(sentence, current_seq)
            if frame is not None:
                if t_first_audio is None:
                    t_first_audio = time.perf_counter()
                emit(frame)
                return current_seq + 1
            return current_seq

        try:
            chunker = SentenceChunker()
            for delta in voice_service.stream_reply(
                request, recent_messages, user_text, addressed, now
            ):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                full_parts.append(delta)
                for sentence in chunker.add(delta):
                    seq = handle_sentence(sentence, seq)
            for sentence in chunker.flush():
                seq = handle_sentence(sentence, seq)

            full_reply = "".join(full_parts).strip()
            if full_reply:
                voice_service.conversations.append(
                    request.guild_id,
                    ConversationMessage(
                        timestamp=now,
                        speaker_id="assistant:butler",
                        speaker_name="Butler",
                        role="assistant",
                        text=full_reply,
                    ),
                )

            def _ms(mark: Optional[float]) -> float:
                return (mark - t_recv) * 1000.0 if mark is not None else -1.0

            logger.info(
                "stream latency guild_id=%s utterance_id=%s first_token_ms=%.0f "
                "first_audio_ms=%.0f total_ms=%.0f chunks=%s reply=%r",
                request.guild_id,
                request.utterance_id,
                _ms(t_first_token),
                _ms(t_first_audio),
                (time.perf_counter() - t_recv) * 1000.0,
                seq,
                full_reply,
            )
            emit({"type": "end", "reply_text": full_reply, "chunks": seq})
        except Exception as error:
            logger.exception(
                "streaming reply failed guild_id=%s utterance_id=%s speaker=%s",
                request.guild_id,
                request.utterance_id,
                speaker_label,
            )
            emit({"type": "error", "message": f"{type(error).__name__}: {error}"})
        finally:
            emit(_STREAM_SENTINEL)

    threading.Thread(target=worker, name="talos-stream", daemon=True).start()

    while True:
        item = await queue.get()
        if item is _STREAM_SENTINEL:
            break
        yield _ndjson(item)


@app.post("/process-audio/stream")
async def process_audio_stream(request: AudioProcessRequest) -> StreamingResponse:
    t_recv = time.perf_counter()
    audio_bytes = base64.b64decode(request.audio_base64)
    speaker_label = (
        request.discord_display_name
        or request.discord_username
        or request.speaker_id
    )
    logger.info(
        "received audio chunk (stream) guild_id=%s utterance_id=%s speaker_id=%s display_name=%s participants=%s bytes=%s",
        request.guild_id,
        request.utterance_id,
        request.speaker_id,
        speaker_label,
        len(request.users_in_call),
        len(audio_bytes),
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        wav_path = Path(temp_file.name)
        temp_file.write(audio_bytes)

    t_stt = time.perf_counter()
    try:
        transcript = await asyncio.to_thread(voice_service.transcribe_wav, wav_path)
    finally:
        wav_path.unlink(missing_ok=True)
    logger.info(
        "stt latency (stream) utterance_id=%s stt_ms=%.0f transcript_chars=%s",
        request.utterance_id,
        (time.perf_counter() - t_stt) * 1000.0,
        len(transcript),
    )

    def _meta_only(should_respond: bool, ignore_reason: Optional[str]) -> StreamingResponse:
        async def _single() -> AsyncIterator[bytes]:
            yield _ndjson(
                {
                    "type": "meta",
                    "transcript": transcript,
                    "should_respond": should_respond,
                    "ignore_reason": ignore_reason,
                }
            )

        return StreamingResponse(_single(), media_type="application/x-ndjson")

    if not transcript:
        logger.info(
            "ignoring (stream) utterance_id=%s speaker=%s reason=empty_transcript",
            request.utterance_id,
            speaker_label,
        )
        return _meta_only(False, "empty_transcript")

    now = datetime.now().astimezone()
    participant_count = len(request.users_in_call)
    requires_wake_word = wake_word_required(participant_count)
    addressed = is_addressed(transcript) if requires_wake_word else True
    recent_messages = voice_service.conversations.recent_messages(request.guild_id)
    user_text = strip_wake_words(transcript) if addressed else transcript

    voice_service.conversations.append(
        request.guild_id,
        ConversationMessage(
            timestamp=now,
            speaker_id=request.speaker_id,
            speaker_name=speaker_label,
            role="user",
            text=user_text,
        ),
    )
    logger.info(
        "transcript analyzed (stream) utterance_id=%s speaker=%s transcript=%r participants=%s requires_wake_word=%s addressed=%s",
        request.utterance_id,
        speaker_label,
        transcript,
        participant_count,
        requires_wake_word,
        addressed,
    )

    if not addressed:
        logger.info(
            "ignoring (stream) utterance_id=%s speaker=%s reason=not_addressed",
            request.utterance_id,
            speaker_label,
        )
        return _meta_only(False, "not_addressed")

    return StreamingResponse(
        _reply_stream(
            request,
            recent_messages,
            user_text,
            addressed,
            now,
            transcript,
            speaker_label,
            t_recv,
        ),
        media_type="application/x-ndjson",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
