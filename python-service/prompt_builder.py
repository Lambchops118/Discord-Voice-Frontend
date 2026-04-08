from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from conversation_store import ConversationMessage
from prompt_config import SYSTEM_PROMPT


@dataclass(slots=True)
class PromptParticipant:
    discord_user_id: int | None
    username: str | None
    display_name: str | None


@dataclass(slots=True)
class PromptSpeaker:
    speaker_id: str
    discord_user_id: int | None
    username: str | None
    display_name: str | None


def build_prompt_context(
    *,
    guild_id: int,
    guild_name: str | None,
    voice_channel_id: int | None,
    voice_channel_name: str | None,
    now: datetime,
    current_speaker: PromptSpeaker,
    participants: Sequence[PromptParticipant],
    recent_messages: Sequence[ConversationMessage],
    latest_user_utterance: str,
    addressed: bool,
) -> dict[str, Any]:
    return {
        "environment": {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S %Z"),
            "current_timestamp": now.isoformat(),
            "guild_id": guild_id,
            "guild_name": guild_name,
            "voice_channel_id": voice_channel_id,
            "voice_channel_name": voice_channel_name,
        },
        "participants": [asdict(participant) for participant in participants],
        "current_speaker": asdict(current_speaker),
        "addressed": addressed,
        "recent_messages": [
            {
                "timestamp": message.timestamp.isoformat(),
                "speaker_id": message.speaker_id,
                "speaker_name": message.speaker_name,
                "role": message.role,
                "text": message.text,
            }
            for message in recent_messages
        ],
        "latest_user_utterance": latest_user_utterance,
    }


def format_recent_history(
    messages: Sequence[ConversationMessage | dict[str, Any]]
) -> str:
    if not messages:
        return "No recent conversation history."

    lines = []
    for message in messages:
        if isinstance(message, ConversationMessage):
            timestamp = message.timestamp.strftime("%H:%M:%S")
            role = message.role
            speaker_name = message.speaker_name
            speaker_id = message.speaker_id
            text = message.text
        else:
            timestamp = datetime.fromisoformat(message["timestamp"]).strftime("%H:%M:%S")
            role = message["role"]
            speaker_name = message["speaker_name"]
            speaker_id = message["speaker_id"]
            text = message["text"]
        lines.append(
            f"[{timestamp}] {role} | {speaker_name} ({speaker_id}): {text}"
        )
    return "\n".join(lines)


def build_model_input(context: dict[str, Any]) -> list[dict[str, Any]]:
    structured_context = {
        "environment": context["environment"],
        "participants": context["participants"],
        "current_speaker": context["current_speaker"],
        "addressed": context["addressed"],
    }
    history = format_recent_history(context["recent_messages"])
    latest_user_utterance = context["latest_user_utterance"]

    user_text = "\n\n".join(
        [
            "STRUCTURED_CONTEXT_JSON",
            json.dumps(structured_context, indent=2, sort_keys=True),
            "RECENT_CONVERSATION",
            history,
            "LATEST_UTTERANCE",
            latest_user_utterance,
            "TASK",
            "Reply to the latest utterance as Butler in this live voice conversation.",
        ]
    )

    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        },
    ]
