from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Literal


Role = Literal["user", "assistant"]


@dataclass(slots=True)
class ConversationMessage:
    timestamp: datetime
    speaker_id: str
    speaker_name: str
    role: Role
    text: str


class ConversationStore:
    def __init__(self, max_history_messages: int) -> None:
        self.max_history_messages = max_history_messages
        self._messages: dict[int, list[ConversationMessage]] = defaultdict(list)
        self._lock = Lock()

    def append(self, guild_id: int, message: ConversationMessage) -> None:
        with self._lock:
            history = self._messages[guild_id]
            history.append(message)
            if len(history) > self.max_history_messages:
                del history[:-self.max_history_messages]

    def recent_messages(self, guild_id: int) -> list[ConversationMessage]:
        with self._lock:
            return list(self._messages.get(guild_id, ()))
