from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Iterator, Protocol

from openai import APIError, OpenAI


logger = logging.getLogger("python-service.llm")


class LLMClient(Protocol):
    def generate_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> str:
        ...

    def stream_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> Iterator[str]:
        """Yield incremental text deltas for the reply.

        Clients that cannot stream natively (e.g. Grok) satisfy this by yielding
        the whole reply as a single delta, so callers get identical output shape.
        """
        ...


@dataclass
class GrokConfig:
    api_key: str | None
    model: str
    base_url: str | None
    timeout_seconds: float
    max_output_tokens: int


class GrokLLMClient:
    def __init__(self, config: GrokConfig) -> None:
        self.config = config
        self.enabled = bool(config.api_key)
        self.client = (
            OpenAI(
                api_key=config.api_key,
                base_url=config.base_url or None,
                timeout=config.timeout_seconds,
                max_retries=0,
            )
            if self.enabled
            else None
        )

    def generate_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> str:
        # session_id / user_text are part of the shared LLMClient contract but the
        # Grok path carries all of its context inside ``model_input``.
        if not self.enabled or self.client is None:
            raise RuntimeError("XAI_API_KEY is not configured")

        try:
            response = self.client.responses.create(
                model=self.config.model,
                input=model_input,
                max_output_tokens=self.config.max_output_tokens,
            )
        except APIError:
            logger.exception("grok responses.create failed")
            raise
        except Exception:
            logger.exception("unexpected grok client failure")
            raise

        reply_text = getattr(response, "output_text", "") or ""
        reply_text = reply_text.strip()
        if not reply_text:
            raise RuntimeError("Grok returned an empty reply")

        return reply_text

    def stream_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> Iterator[str]:
        # Grok's Responses call is one-shot, so we preserve existing behavior:
        # generate the full reply, then hand it off as a single delta for the
        # caller's sentence chunker to split and synthesize.
        yield self.generate_reply(
            model_input, session_id=session_id, user_text=user_text
        )


@dataclass
class TalosConfig:
    """Connection details for the TALOS text agent (`POST /chat`).

    TALOS owns its own personality, memory, tool routing, and conversation
    history server-side, so this client sends only the latest user utterance and
    a stable ``session_id``. The heavier prompt-building/history machinery used by
    the Grok path is intentionally bypassed.
    """

    base_url: str
    token: str
    source: str
    mode: str
    timeout_seconds: float
    # Relative path of the token-streaming SSE endpoint (POST). Mirrors
    # talos/text/service_client.py:stream_message.
    stream_path: str = "/chat/stream"


class TalosLLMClient:
    """Routes replies to the TALOS brain instead of a hosted LLM.

    Mirrors the request shape of ``talos/text/service_client.py`` (the same
    endpoint the TALOS GUI and voice worker use) but depends only on the standard
    library so the Discord service gains no new packages.
    """

    def __init__(self, config: TalosConfig) -> None:
        self.config = config
        # No API key is required: reaching the local text agent is always allowed.
        # Readiness is a runtime concern (the agent may not be up yet), surfaced
        # as a normal request error rather than a disabled client.
        self.enabled = True

    def _chat(self, message: str, session_id: str) -> str:
        payload = {
            "message": message,
            "session_id": session_id or self.config.source,
            "source": self.config.source,
            "mode": self.config.mode,
        }
        data = json.dumps(payload).encode("utf-8")
        url = urllib.parse.urljoin(
            self.config.base_url.rstrip("/") + "/", "chat"
        )
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        if self.config.token:
            request.add_header("Authorization", f"Bearer {self.config.token}")

        timeout = self.config.timeout_seconds if self.config.timeout_seconds > 0 else None
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"TALOS HTTP {exc.code}: {body_text or exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TALOS connection error: {exc.reason}") from exc

        body = json.loads(raw) if raw else {}
        if not isinstance(body, dict):
            raise RuntimeError("TALOS returned a non-object JSON payload")
        if not body.get("ok", True):
            raise RuntimeError(body.get("error", "TALOS returned an error"))
        reply_text = str(body.get("response", "")).strip()
        if not reply_text:
            raise RuntimeError("TALOS returned an empty reply")
        return reply_text

    def generate_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> str:
        # ``model_input`` is ignored: TALOS builds its own prompt. Fall back to
        # extracting the last user string only if ``user_text`` was not supplied.
        message = user_text.strip()
        if not message:
            message = _latest_user_text(model_input)
        if not message:
            raise RuntimeError("No user utterance to send to TALOS")
        return self._chat(message, session_id)

    def stream_reply(
        self,
        model_input: list[dict[str, object]],
        *,
        session_id: str = "",
        user_text: str = "",
    ) -> Iterator[str]:
        message = user_text.strip()
        if not message:
            message = _latest_user_text(model_input)
        if not message:
            raise RuntimeError("No user utterance to send to TALOS")
        yield from self._chat_stream(message, session_id)

    def _chat_stream(self, message: str, session_id: str) -> Iterator[str]:
        """Consume TALOS's ``POST /chat/stream`` SSE endpoint, yielding text deltas.

        The SSE framing is standard (``data:`` lines terminated by a blank line);
        the delta field name varies between agent builds, so ``_extract_delta``
        accepts the common shapes. Falls back to the blocking ``/chat`` endpoint
        if streaming is unavailable so a reply is still produced.
        """
        payload = {
            "message": message,
            "session_id": session_id or self.config.source,
            "source": self.config.source,
            "mode": self.config.mode,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        url = urllib.parse.urljoin(
            self.config.base_url.rstrip("/") + "/",
            self.config.stream_path.lstrip("/"),
        )
        request = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            method="POST",
        )
        if self.config.token:
            request.add_header("Authorization", f"Bearer {self.config.token}")

        timeout = (
            self.config.timeout_seconds if self.config.timeout_seconds > 0 else None
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            # 404/405 → this agent build has no streaming endpoint; degrade to
            # the blocking call rather than failing the turn.
            if exc.code in (404, 405, 501):
                logger.warning(
                    "TALOS stream endpoint unavailable (HTTP %s); falling back to /chat",
                    exc.code,
                )
                yield self._chat(message, session_id)
                return
            raise RuntimeError(f"TALOS stream HTTP {exc.code}: {body_text or exc}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"TALOS stream connection error: {exc.reason}") from exc

        emitted_any = False
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data_part = line[len("data:") :].strip()
                if not data_part:
                    continue
                if data_part.upper() == "[DONE]":
                    break

                try:
                    event = json.loads(data_part)
                except json.JSONDecodeError:
                    # Non-JSON line: treat the raw payload as a delta.
                    emitted_any = True
                    yield data_part
                    continue

                event_type = event.get("type") if isinstance(event, dict) else None
                if event_type == "delta":
                    # TALOS protocol: only ``delta`` events carry incremental
                    # text. The terminal ``done`` event ALSO includes the full
                    # ``text`` — extracting it here is what caused the reply to
                    # play twice, so we intentionally ignore ``done``'s text.
                    text = event.get("text") or ""
                    if text:
                        emitted_any = True
                        yield text
                elif event_type == "done":
                    break
                elif event_type == "error":
                    raise RuntimeError(str(event.get("error") or "TALOS stream error"))
                elif event_type is None:
                    # Unknown schema (e.g. OpenAI-style): best-effort extraction.
                    delta = _extract_delta(event)
                    if delta:
                        emitted_any = True
                        yield delta
                # Any other typed event is a control frame we don't need.
        finally:
            response.close()

        if not emitted_any:
            # The stream connected but yielded nothing usable (unexpected schema).
            # Fall back so the user still hears a reply this turn.
            logger.warning(
                "TALOS stream produced no text deltas; falling back to /chat"
            )
            yield self._chat(message, session_id)


def _extract_delta(obj: object) -> str:
    """Best-effort extraction of a text delta from a parsed SSE payload with no
    ``type`` field (e.g. an OpenAI-style event). Used only as a fallback for
    non-TALOS servers; the TALOS protocol is handled explicitly by ``type``.
    """
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return ""
    # OpenAI-style: {"choices": [{"delta": {"content": "..."}}]}
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                text = delta.get("content") or delta.get("text")
                if isinstance(text, str):
                    return text
            text = first.get("text")
            if isinstance(text, str):
                return text
    for key in ("delta", "token", "content", "text", "chunk", "response"):
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return ""


def _latest_user_text(model_input: list[dict[str, object]]) -> str:
    """Best-effort extraction of the newest user utterance from an OpenAI-style
    ``input`` list, used only as a fallback when ``user_text`` is absent."""

    for item in reversed(model_input or []):
        if not isinstance(item, dict):
            continue
        if item.get("role") not in (None, "user"):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return ""
