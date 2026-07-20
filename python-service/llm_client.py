from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

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
