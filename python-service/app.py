import asyncio
import base64
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import boto3
import uvicorn
from conversation_store import ConversationMessage, ConversationStore
from dotenv import load_dotenv
from fastapi import FastAPI
from faster_whisper import WhisperModel
from llm_client import OpenAIConfig, OpenAILLMClient
from logic import is_addressed, strip_wake_words
from pydantic import BaseModel, Field
from prompt_builder import (
    PromptParticipant,
    PromptSpeaker,
    build_model_input,
    build_prompt_context,
)

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
    aws_region: str = os.getenv(
        "AWS_REGION",
        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )
    polly_voice_id: str = os.getenv("POLLY_VOICE_ID", "Brian")
    polly_engine: str = os.getenv("POLLY_ENGINE", "neural")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")
    openai_timeout_seconds: float = float(
        os.getenv("OPENAI_TIMEOUT_SECONDS", "20")
    )
    openai_max_output_tokens: int = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "180"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY_MESSAGES", "24"))


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


class HealthResponse(BaseModel):
    status: str
    whisper_model: str


class VoiceService:
    def __init__(self, config: Settings) -> None:
        self.config = config
        logger.info(
            "loading faster-whisper model=%s device=%s compute_type=%s",
            config.whisper_model,
            config.whisper_device,
            config.whisper_compute_type,
        )
        self.model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )
        logger.info("faster-whisper model loaded")
        self.polly = boto3.client("polly", region_name=config.aws_region)
        logger.info(
            "configured aws polly region=%s voice=%s engine=%s",
            config.aws_region,
            config.polly_voice_id,
            config.polly_engine,
        )
        self.llm = OpenAILLMClient(
            OpenAIConfig(
                api_key=config.openai_api_key,
                model=config.openai_model,
                base_url=config.openai_base_url,
                timeout_seconds=config.openai_timeout_seconds,
                max_output_tokens=config.openai_max_output_tokens,
            )
        )
        self.conversations = ConversationStore(
            max_history_messages=config.max_history_messages,
        )
        logger.info(
            "configured llm provider=openai enabled=%s model=%s timeout_seconds=%s max_history_messages=%s",
            self.llm.enabled,
            config.openai_model,
            config.openai_timeout_seconds,
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

    def synthesize_tts_sync(self, text: str) -> bytes:
        response = self.polly.synthesize_speech(
            Engine=self.config.polly_engine,
            OutputFormat="mp3",
            SampleRate="24000",
            Text=text,
            TextType="text",
            VoiceId=self.config.polly_voice_id,
        )
        audio_stream = response["AudioStream"]
        try:
            audio_bytes = audio_stream.read()
            logger.info("tts generated bytes=%s", len(audio_bytes))
            return audio_bytes
        finally:
            audio_stream.close()

    async def synthesize_tts(self, text: str) -> bytes:
        return await asyncio.to_thread(self.synthesize_tts_sync, text)

    def generate_reply(
        self,
        request: AudioProcessRequest,
        recent_messages: list[ConversationMessage],
        user_text: str,
        addressed: bool,
        now: datetime,
    ) -> str:
        speaker_name = (
            request.discord_display_name
            or request.discord_username
            or request.speaker_id
        )
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
            "calling llm guild_id=%s utterance_id=%s speaker=%s participants=%s history_messages=%s addressed=%s",
            request.guild_id,
            request.utterance_id,
            speaker_name,
            len(request.users_in_call),
            len(recent_messages),
            addressed,
        )
        return self.llm.generate_reply(model_input)


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
    addressed = is_addressed(transcript)
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
        "transcript analyzed utterance_id=%s speaker=%s transcript=%r addressed=%s should_attempt_reply=%s",
        request.utterance_id,
        speaker_label,
        transcript,
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
            tts_audio = await voice_service.synthesize_tts(fallback_reply)
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
            tts_audio_format="mp3",
        )

    try:
        tts_audio = await voice_service.synthesize_tts(reply_text)
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
        tts_audio_format="mp3",
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
