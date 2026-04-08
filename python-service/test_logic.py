import unittest

from logic import choose_reply, is_addressed


class WakeWordTests(unittest.TestCase):
    def test_detects_butler_anywhere(self) -> None:
        self.assertTrue(is_addressed("Can you help, Butler?"))

    def test_detects_monkey_butler_phrase(self) -> None:
        self.assertTrue(is_addressed("hey monkey butler, what time is it"))

    def test_detects_monkey_with_punctuation(self) -> None:
        self.assertTrue(is_addressed("monkey... say hello"))

    def test_rejects_partial_word_match(self) -> None:
        self.assertFalse(is_addressed("the butlers are busy"))

    def test_rejects_unaddressed_transcript(self) -> None:
        self.assertFalse(is_addressed("this is just normal channel chatter"))

    def test_reports_current_speaker_name(self) -> None:
        self.assertEqual(
            choose_reply("butler who is speaking", "Alice"),
            "Alice is speaking.",
        )

    def test_reports_unknown_current_speaker(self) -> None:
        self.assertEqual(
            choose_reply("butler who's speaking", None),
            "I can't tell who is speaking yet.",
        )


if __name__ == "__main__":
    unittest.main()
