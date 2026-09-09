import pytest

from src.tokenizer import BPETokenizer


@pytest.fixture
def tokenizer():
    return BPETokenizer(vocab_size=10)


def test_compute_word_freqs_counts_occurrences(tokenizer):
    freqs = tokenizer.compute_word_freqs(["the", "cat", "sat", "the"])
    assert freqs["the"] == 2
    assert freqs["cat"] == 1
    assert freqs["sat"] == 1


def test_compute_splits_returns_characters_per_word(tokenizer):
    splits = tokenizer.compute_splits({"the": 2, "cat": 1})
    assert splits == {"the": ["Ġ", "t", "h", "e"], "cat": ["Ġ", "c", "a", "t"]}


def test_compute_pair_freqs_counts_adjacent_pairs(tokenizer):
    word_freqs = {"the": 2, "cat": 1}
    splits = {"the": ["t", "h", "e"], "cat": ["c", "a", "t"]}

    pair_freqs = tokenizer.compute_pair_freqs(splits, word_freqs)

    assert pair_freqs[("t", "h")] == 2
    assert pair_freqs[("h", "e")] == 2
    assert pair_freqs[("c", "a")] == 1
    assert pair_freqs[("a", "t")] == 1


def test_merge_pair_merges_adjacent_tokens(tokenizer):
    splits = {"the": ["t", "h", "e"], "cat": ["c", "a", "t"]}
    word_freqs = {"the": 2, "cat": 1}

    new_splits = tokenizer.merge_pair("t", "h", splits, word_freqs)

    assert new_splits["the"] == ["th", "e"]
    assert new_splits["cat"] == ["c", "a", "t"]


def test_compute_alphabet_returns_unique_characters(tokenizer):
    alphabet = tokenizer.compute_alphabet({"the": 2, "cat": 1})
    assert alphabet == sorted(set("thecat"))


def test_train_builds_vocab_up_to_target_size():
    tokenizer = BPETokenizer(vocab_size=7)
    tokenizer.train(["ab", "ab", "ab", "cd"])

    assert len(tokenizer.str_to_int) == 7
    assert "Ġab" in tokenizer.str_to_int


def test_encode_decode_round_trip():
    tokenizer = BPETokenizer(vocab_size=7)
    tokenizer.train(["ab", "ab", "ab", "cd"])

    ids = tokenizer.encode("ab cd")

    assert tokenizer.decode(ids) == "ab cd"
