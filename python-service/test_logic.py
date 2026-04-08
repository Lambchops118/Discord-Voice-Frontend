import unittest

from datetime import datetime, timedelta, timezone

from conversation_store import ConversationMessage, ConversationStore
from logic import is_addressed, strip_wake_words
from prompt_builder import (
    PromptParticipant,
    PromptSpeaker,
    build_model_input,
    build_prompt_context,
    format_recent_history,
)


class WakeWordTests(unittest.TestCase):
    def test_detects_butler_anywhere(self) -> None:
        self.assertTrue(is_addressed("Can you help, Butler?"))

    def test_detects_monkey_butler_phrase(self) -> None:
        self.assertTrue(is_addressed("hey monkey butler, what time is it"))

    def test_detects_monkey_with_punctuation(self) -> None:
        self.assertTrue(is_addressed("monkey... say hello"))

    def test_detects_clanker(self) -> None:
        self.assertTrue(is_addressed("clanker, are you there"))

    def test_rejects_partial_word_match(self) -> None:
        self.assertFalse(is_addressed("the butlers are busy"))

    def test_rejects_unaddressed_transcript(self) -> None:
        self.assertFalse(is_addressed("this is just normal channel chatter"))

    def test_strips_wake_words_from_latest_utterance(self) -> None:
        self.assertEqual(
            strip_wake_words("butler, what time is it"),
            "what time is it",
        )


class ConversationStoreTests(unittest.TestCase):
    def test_trims_history_to_max_messages(self) -> None:
        store = ConversationStore(max_history_messages=2)
        now = datetime.now(timezone.utc)
        store.append(
            1,
            ConversationMessage(
                timestamp=now,
                speaker_id="user:1",
                speaker_name="Alice",
                role="user",
                text="one",
            ),
        )
        store.append(
            1,
            ConversationMessage(
                timestamp=now,
                speaker_id="assistant:butler",
                speaker_name="Butler",
                role="assistant",
                text="two",
            ),
        )
        store.append(
            1,
            ConversationMessage(
                timestamp=now,
                speaker_id="user:2",
                speaker_name="Bob",
                role="user",
                text="three",
            ),
        )

        self.assertEqual(
            [message.text for message in store.recent_messages(1)],
            ["two", "three"],
        )


class PromptBuilderTests(unittest.TestCase):
    def test_formats_speaker_attributed_history(self) -> None:
        now = datetime(2026, 4, 8, 12, 1, 5, tzinfo=timezone.utc)
        history = [
            ConversationMessage(
                timestamp=now,
                speaker_id="discord:1",
                speaker_name="Alice",
                role="user",
                text="what time is it",
            ),
            ConversationMessage(
                timestamp=now,
                speaker_id="assistant:butler",
                speaker_name="Butler",
                role="assistant",
                text="It's 8:01 AM.",
            ),
        ]

        formatted = format_recent_history(history)

        self.assertIn("Alice (discord:1): what time is it", formatted)
        self.assertIn("Butler (assistant:butler): It's 8:01 AM.", formatted)

    def test_prompt_input_contains_context_sections(self) -> None:
        now = datetime(2026, 4, 8, 12, 1, 5, tzinfo=timezone.utc)
        context = build_prompt_context(
            guild_id=123,
            guild_name="Test Guild",
            voice_channel_id=456,
            voice_channel_name="General",
            now=now,
            current_speaker=PromptSpeaker(
                speaker_id="discord:1",
                discord_user_id=1,
                username="alice",
                display_name="Alice",
            ),
            participants=[
                PromptParticipant(
                    discord_user_id=1,
                    username="alice",
                    display_name="Alice",
                ),
                PromptParticipant(
                    discord_user_id=2,
                    username="bob",
                    display_name="Bob",
                ),
            ],
            recent_messages=[],
            latest_user_utterance="Can you help Bob with that?",
            addressed=True,
        )

        model_input = build_model_input(context)
        user_content = model_input[1]["content"][0]["text"]

        self.assertIn("STRUCTURED_CONTEXT_JSON", user_content)
        self.assertIn("RECENT_CONVERSATION", user_content)
        self.assertIn("LATEST_UTTERANCE", user_content)
        self.assertIn("Test Guild", user_content)
        self.assertIn("Can you help Bob with that?", user_content)


if __name__ == "__main__":
    unittest.main()
