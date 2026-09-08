import unittest

from src.vocabBuild import (
    build_wordpiece_vocab,
    compute_pair_stats,
    get_best_pair,
    initial_wordpiece_split,
    merge_pair,
    merge_tokens,
)


class TestVocabBuild(unittest.TestCase):
    def test_initial_wordpiece_split_word(self):
        result = initial_wordpiece_split("play")

        self.assertEqual(result, ["p", "##l", "##a", "##y"])

    def test_initial_wordpiece_split_single_character(self):
        result = initial_wordpiece_split("a")

        self.assertEqual(result, ["a"])

    def test_initial_wordpiece_split_punctuation(self):
        result = initial_wordpiece_split("!")

        self.assertEqual(result, ["!"])

    def test_compute_pair_stats(self):
        splits = {"play": ["p", "##l", "##a", "##y"]}

        token_freqs = {"play": 3}

        pair_freqs, piece_freqs = compute_pair_stats(splits, token_freqs)

        self.assertEqual(piece_freqs["p"], 3)
        self.assertEqual(piece_freqs["##l"], 3)

        self.assertEqual(pair_freqs[("p", "##l")], 3)

    def test_get_best_pair(self):
        splits = {"ab": ["a", "##b"], "ac": ["a", "##c"]}

        token_freqs = {"ab": 5, "ac": 1}

        best_pair = get_best_pair(splits, token_freqs)

        self.assertIn(best_pair, [("a", "##b"), ("a", "##c")])

    def test_merge_tokens_first_piece(self):
        result = merge_tokens("p", "##l")

        self.assertEqual(result, "pl")

    def test_merge_tokens_continuation_piece(self):
        result = merge_tokens("##a", "##y")

        self.assertEqual(result, "##ay")

    def test_merge_pair(self):
        splits = {"play": ["p", "##l", "##a", "##y"]}

        new_splits, merged = merge_pair(("p", "##l"), splits)

        self.assertEqual(merged, "pl")

        self.assertEqual(new_splits["play"], ["pl", "##a", "##y"])

    def test_build_vocab_contains_special_tokens(self):
        preprocessed = ["play", "playing", "played"]

        vocab = build_wordpiece_vocab(preprocessed, target_vocab_size=50)

        self.assertIn("<|unk|>", vocab)

        self.assertIn("<|endoftext|>", vocab)

    def test_build_vocab_returns_integer_ids(self):
        preprocessed = ["hello", "world"]

        vocab = build_wordpiece_vocab(preprocessed, target_vocab_size=50)

        for token, token_id in vocab.items():
            self.assertIsInstance(token, str)

            self.assertIsInstance(token_id, int)

    def test_build_vocab_ids_are_unique(self):
        preprocessed = ["hello", "world", "hello"]

        vocab = build_wordpiece_vocab(preprocessed, target_vocab_size=50)

        ids = list(vocab.values())

        self.assertEqual(len(ids), len(set(ids)))

    def test_build_vocab_does_not_exceed_target_unnecessarily(self):
        preprocessed = ["cat", "cats", "dog", "dogs"]

        target_size = 30

        vocab = build_wordpiece_vocab(preprocessed, target_vocab_size=target_size)

        self.assertLessEqual(len(vocab), target_size)


if __name__ == "__main__":
    unittest.main()
