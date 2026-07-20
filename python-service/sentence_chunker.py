"""Incremental sentence chunking for streamed LLM output.

Mirrors ``talos/voice/streaming/sentence_chunker.py``: tokens arrive one at a
time, and we emit a chunk as soon as a sentence boundary is reached so it can be
synthesized while the rest of the reply is still generating. This is what lets
playback start on the first sentence instead of waiting for the whole reply.
"""

from __future__ import annotations

import re

# End-of-sentence punctuation followed by whitespace (or end of buffer). Kept
# deliberately simple/greedy — mis-splitting on an abbreviation only costs a
# slightly shorter audio chunk, never a dropped word.
_BOUNDARY = re.compile(r"[.!?…]+[\"')\]]*\s")


class SentenceChunker:
    """Accumulates streamed text and yields complete sentences.

    ``min_chars`` avoids emitting tiny fragments (e.g. "Hi.") that would make
    Polly stutter; ``max_chars`` force-flushes a runaway sentence that never
    hits punctuation so audio keeps flowing.
    """

    def __init__(self, min_chars: int = 12, max_chars: int = 240) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buffer = ""

    def add(self, text: str) -> list[str]:
        """Feed a token/delta; return any newly-completed sentences."""
        if not text:
            return []
        self._buffer += text
        return self._drain()

    def flush(self) -> list[str]:
        """Emit whatever is left once the stream ends."""
        remainder = self._buffer.strip()
        self._buffer = ""
        return [remainder] if remainder else []

    def _drain(self) -> list[str]:
        out: list[str] = []
        while True:
            match = _BOUNDARY.search(self._buffer)
            if match:
                end = match.end()
                candidate = self._buffer[:end].strip()
                # Hold short fragments back and let them merge with the next
                # sentence rather than synthesizing a one-word clip.
                if len(candidate) < self.min_chars:
                    # Only keep waiting if more text could still arrive; if the
                    # buffer is already long we fall through to the max-chars
                    # guard below on the next iteration.
                    if len(self._buffer) < self.max_chars:
                        break
                out.append(candidate)
                self._buffer = self._buffer[end:]
                continue
            # No boundary: force-flush an over-long buffer at the last space.
            if len(self._buffer) >= self.max_chars:
                split_at = self._buffer.rfind(" ", 0, self.max_chars)
                if split_at <= 0:
                    split_at = self.max_chars
                candidate = self._buffer[:split_at].strip()
                if candidate:
                    out.append(candidate)
                self._buffer = self._buffer[split_at:]
                continue
            break
        return out
