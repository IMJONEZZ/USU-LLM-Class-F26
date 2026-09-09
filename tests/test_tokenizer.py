import unittest

from src.tokenizer import WordPieceTokenizer


class TestWordPieceTokenizer(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "<|unk|>": 0,
            "<|endoftext|>": 1,
            "play": 2,
            "##ing": 3,
            "##ed": 4,
            "hello": 5,
            "world": 6,
            ",": 7,
            "!": 8,
        }

        self.tokenizer = WordPieceTokenizer(self.vocab)

    def test_tokenize_word_direct_match(self):
        result = self.tokenizer.tokenize_word("play")

        self.assertEqual(result, ["play"])

    def test_tokenize_word_into_subwords(self):
        result = self.tokenizer.tokenize_word("playing")

        self.assertEqual(result, ["play", "##ing"])

    def test_tokenize_word_another_subword(self):
        result = self.tokenizer.tokenize_word("played")

        self.assertEqual(result, ["play", "##ed"])

    def test_tokenize_word_unknown(self):
        result = self.tokenizer.tokenize_word("spaceship")

        self.assertEqual(result, ["<|unk|>"])

    def test_encode_single_word(self):
        result = self.tokenizer.encode("play")

        self.assertEqual(result, [2])

    def test_encode_wordpiece_word(self):
        result = self.tokenizer.encode("playing")

        self.assertEqual(result, [2, 3])

    def test_encode_sentence(self):
        result = self.tokenizer.encode("hello world")

        self.assertEqual(result, [5, 6])

    def test_encode_with_punctuation(self):
        result = self.tokenizer.encode("hello, world!")

        self.assertEqual(result, [5, 7, 6, 8])

    def test_encode_unknown_word(self):
        result = self.tokenizer.encode("hello spaceship")

        self.assertEqual(result, [5, 0])

    def test_decode_single_word(self):
        result = self.tokenizer.decode([2])

        self.assertEqual(result, "play")

    def test_decode_wordpiece_tokens(self):
        result = self.tokenizer.decode([2, 3])

        self.assertEqual(result, "playing")

    def test_decode_sentence(self):
        result = self.tokenizer.decode([5, 6])

        self.assertEqual(result, "hello world")

    def test_decode_punctuation(self):
        result = self.tokenizer.decode([5, 7, 6, 8])

        self.assertEqual(result, "hello, world!")

    def test_round_trip_basic_sentence(self):
        original = "hello, world!"

        ids = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(ids)

        self.assertEqual(decoded, original)

    def test_round_trip_wordpiece(self):
        original = "playing"

        ids = self.tokenizer.encode(original)
        decoded = self.tokenizer.decode(ids)

        self.assertEqual(decoded, original)


if __name__ == "__main__":
    unittest.main()
