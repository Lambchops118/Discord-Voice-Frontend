from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from openai import APIError, OpenAI


logger = logging.getLogger("python-service.llm")


class LLMClient(Protocol):
    def generate_reply(self, model_input: list[dict[str, object]]) -> str:
        ...


@dataclass
class OpenAIConfig:
    api_key: str | None
    model: str
    base_url: str | None
    timeout_seconds: float
    max_output_tokens: int


class OpenAILLMClient:
    def __init__(self, config: OpenAIConfig) -> None:
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

    def generate_reply(self, model_input: list[dict[str, object]]) -> str:
        if not self.enabled or self.client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        try:
            response = self.client.responses.create(
                model=self.config.model,
                input=model_input,
                max_output_tokens=self.config.max_output_tokens,
            )
        except APIError:
            logger.exception("openai responses.create failed")
            raise
        except Exception:
            logger.exception("unexpected openai client failure")
            raise

        reply_text = getattr(response, "output_text", "") or ""
        reply_text = reply_text.strip()
        if not reply_text:
            raise RuntimeError("OpenAI returned an empty reply")

        return reply_text
