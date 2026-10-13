import json

import pytest

from src.bpe_tokenizer import (
    END_OF_WORD,
    BPETokenizer,
    get_pair_counts,
    get_word_counts,
    load_dialogue,
    merge_pair,
    train_bpe,
)

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def training_text():
    """A small repetitive corpus so BPE has clear, predictable merges to learn."""
    return "the cat sat on the mat. the cat ran. the dog sat."


@pytest.fixture
def trained_bpe(training_text):
    """Merges and vocab learned from the small training corpus."""
    merges, vocab = train_bpe(training_text, num_merges=20)
    return merges, vocab


@pytest.fixture
def tokenizer(trained_bpe):
    """A BPETokenizer built from the small trained vocabulary."""
    merges, vocab = trained_bpe
    return BPETokenizer(merges, vocab)


# ---------------------------------------------------------------------------
# get_word_counts() TESTS
# ---------------------------------------------------------------------------


def test_get_word_counts_splits_words():
    """Each word should be represented as a tuple of characters plus end marker."""
    counts = get_word_counts("cat cat dog")
    cat_key = tuple("cat") + (END_OF_WORD,)
    dog_key = tuple("dog") + (END_OF_WORD,)
    assert counts[cat_key] == 2
    assert counts[dog_key] == 1


def test_get_word_counts_empty_string():
    """An empty string should produce no words."""
    counts = get_word_counts("")
    assert counts == {}


# ---------------------------------------------------------------------------
# get_pair_counts() TESTS
# ---------------------------------------------------------------------------


def test_get_pair_counts_counts_adjacent_pairs():
    """Adjacent character pairs should be counted correctly across all words."""
    word_counts = get_word_counts("aa")
    pair_counts = get_pair_counts(word_counts)
    # "aa" + end marker -> ('a','a'), ('a','</w>')
    assert pair_counts[("a", "a")] == 1
    assert pair_counts[("a", END_OF_WORD)] == 1


def test_get_pair_counts_empty_input():
    """No words should mean no pairs."""
    assert get_pair_counts({}) == {}


# ---------------------------------------------------------------------------
# merge_pair() TESTS
# ---------------------------------------------------------------------------


def test_merge_pair_combines_symbols():
    """Merging a pair should replace both symbols with a single combined one."""
    word_counts = {("c", "a", "t", END_OF_WORD): 1}
    merged = merge_pair(("c", "a"), word_counts)
    assert ("ca", "t", END_OF_WORD) in merged


def test_merge_pair_preserves_counts():
    """Merging shouldn't change the frequency count of a word."""
    word_counts = {("c", "a", "t", END_OF_WORD): 5}
    merged = merge_pair(("c", "a"), word_counts)
    assert merged[("ca", "t", END_OF_WORD)] == 5


# ---------------------------------------------------------------------------
# train_bpe() TESTS
# ---------------------------------------------------------------------------


def test_train_bpe_learns_merges(training_text):
    """Training on repetitive text should learn at least one merge."""
    merges, _vocab = train_bpe(training_text, num_merges=20)
    assert len(merges) > 0


def test_train_bpe_respects_num_merges_limit(training_text):
    """Training should never learn more merges than requested."""
    merges, _vocab = train_bpe(training_text, num_merges=5)
    assert len(merges) <= 5


def test_train_bpe_vocab_contains_all_base_characters(training_text):
    """
    Every individual character in the training text must remain in the vocab,
    even if it always gets merged into something else during training.
    This is critical -- without it, encoding an unseen word containing that
    character would have no valid fallback.
    """
    _merges, vocab = train_bpe(training_text, num_merges=20)
    for char in set(training_text.replace(" ", "")):
        assert char in vocab


def test_train_bpe_with_zero_merges_returns_base_vocab_only(training_text):
    """With num_merges=0, no merges should be learned, only base characters exist."""
    merges, vocab = train_bpe(training_text, num_merges=0)
    assert merges == []
    for char in set(training_text.replace(" ", "")):
        assert char in vocab


# ---------------------------------------------------------------------------
# BPETokenizer ENCODE/DECODE TESTS
# ---------------------------------------------------------------------------


def test_encode_returns_list_of_ints(tokenizer):
    """encode() should return a list of integer token IDs."""
    ids = tokenizer.encode("the cat")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_round_trip_known_text(tokenizer):
    """Encoding then decoding known training text should reconstruct it exactly."""
    text = "the cat sat"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_encode_unseen_word_does_not_use_unk(tokenizer):
    """
    The key advantage of BPE over whole-word tokenization: a word never seen
    during training ("hat") should still decode correctly, using learned
    subword pieces and base characters instead of falling back to <|unk|>.
    """
    ids = tokenizer.encode("the hat")
    unk_id = tokenizer.str_to_int["<|unk|>"]
    assert unk_id not in ids


def test_encode_unseen_word_round_trips_correctly(tokenizer):
    """An unseen word should still decode back to itself via subword pieces."""
    ids = tokenizer.encode("the hat")
    decoded = tokenizer.decode(ids)
    assert decoded == "the hat"


def test_encode_empty_string_returns_empty_list(tokenizer):
    """Encoding an empty string should return an empty list, not error."""
    ids = tokenizer.encode("")
    assert ids == []


def test_decode_empty_list_returns_empty_string(tokenizer):
    """Decoding an empty list should return an empty string."""
    assert tokenizer.decode([]) == ""


# ---------------------------------------------------------------------------
# VOCAB MAPPING TESTS
# ---------------------------------------------------------------------------


def test_str_to_int_and_int_to_str_are_inverses(tokenizer):
    """int_to_str should be the exact reverse mapping of str_to_int."""
    for token, idx in tokenizer.str_to_int.items():
        assert tokenizer.int_to_str[idx] == token


def test_special_tokens_present_in_vocab(tokenizer):
    """<|endoftext|> and <|unk|> should always be present in the vocab."""
    assert "<|endoftext|>" in tokenizer.str_to_int
    assert "<|unk|>" in tokenizer.str_to_int


# ---------------------------------------------------------------------------
# load_dialogue() TEST
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_load_dialogue_extracts_lines(tmp_path):
    """load_dialogue should read a JSON file and join all 'Line' fields."""
    fake_data = [
        {"Character": "LUKE", "Line": "Hello there."},
        {"Character": "LEIA", "Line": "General Kenobi."},
    ]
    fake_file = tmp_path / "fake_dataset.json"
    fake_file.write_text(json.dumps(fake_data), encoding="utf-8")

    result = load_dialogue(str(fake_file))

    assert "Hello there." in result
    assert "General Kenobi." in result
