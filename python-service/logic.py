import re

WAKE_PATTERN = re.compile(
    r"\b(?:monkey\s+butler|monkey|butler|clanker)\b",
    re.IGNORECASE,
)


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
