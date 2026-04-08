import re
from datetime import datetime

WAKE_PATTERN = re.compile(r"\b(?:monkey\s+butler|monkey|butler)\b", re.IGNORECASE)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_addressed(transcript: str) -> bool:
    normalized = normalize_text(transcript)
    if not normalized:
        return False

    return WAKE_PATTERN.search(normalized) is not None


def strip_wake_words(transcript: str) -> str:
    stripped = WAKE_PATTERN.sub(" ", transcript, count=1)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,.!?:;-")
    return stripped or transcript.strip()


def choose_reply(transcript: str, speaker_name: str | None = None) -> str:
    content = strip_wake_words(transcript)
    normalized = normalize_text(content)

    if normalized in {"hello", "hello bot", "hi", "hey"} or "hello bot" in normalized:
        return "Hello! Voice pipeline is working."
    if normalized in {"who is speaking", "who s speaking"}:
        if speaker_name:
            return f"{speaker_name} is speaking."
        return "I can't tell who is speaking yet."
    if "what time is it" in normalized:
        current_time = datetime.now().strftime("%I:%M %p").lstrip("0")
        return f"It is {current_time}."
    if normalized.startswith("say "):
        requested = content[4:].strip()
        return f"Test successful. You asked me to say: {requested}"
    if "test successful" in normalized:
        return "Voice test successful."

    return f"I heard you say: {content}"
