import unittest

from logic import is_addressed


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


if __name__ == "__main__":
    unittest.main()
