import json
from unittest.mock import mock_open, patch

import pytest

from src.tokenizer import SentencePieceTokenizer, SimpleTokenizer, process_text

# --- Fixtures ---


@pytest.fixture
def sample_vocab():
    """Provides a small, controlled vocabulary for testing."""
    tokens = [
        "Falcon",
        "Take",
        "Tatooine",
        "back",
        "the",
        "to",
        ".",
        ",",
        "!",
        "?",
        "<|endoftext|>",
        "<|unk|>",
    ]
    return {token: i for i, token in enumerate(tokens)}


@pytest.fixture
def tokenizer(sample_vocab):
    """Provides an initialized SimpleTokenizer instance."""
    return SimpleTokenizer(sample_vocab)


@pytest.fixture
def mock_dataset():
    """Provides sample JSON data matching the assignment's SW format."""
    return [
        {"Line": "Take the Falcon back to Tatooine."},
        {"Line": "We must return to Coruscant!"},
    ]


@pytest.fixture
def sp_tokenizer():
    """Create a trained SentencePieceTokenizer for testing."""
    sp_tokenizer = SentencePieceTokenizer()
    sp_tokenizer.train("hello hello world world", num_merges=10)
    return sp_tokenizer


@pytest.fixture
def untrained_sp_tokenizer():
    """Create an untrained SentencePieceTokenizer."""
    return SentencePieceTokenizer()


# --- Tests for process_text ---


def test_process_text_basic_split():
    text = "Hello, world! This is a test."
    expected = ["Hello", ",", "world", "!", "This", "is", "a", "test", "."]
    assert process_text(text) == expected


def test_process_text_special_delimiters():
    text = "wait--what? 'yes' (no) [skip]"
    expected = ["wait", "--", "what", "?", "'", "yes", "'", "(", "no", ")", "[skip]"]
    assert process_text(text) == expected


def test_process_text_whitespace_and_empty():
    assert process_text("   \n\t  ") == []
    assert process_text("") == []


# --- Tests for SimpleTokenizer ---


def test_tokenizer_initialization(sample_vocab, tokenizer):
    assert tokenizer.str_to_int == sample_vocab
    assert tokenizer.int_to_str[sample_vocab["Falcon"]] == "Falcon"
    assert len(tokenizer.int_to_str) == len(sample_vocab)


def test_encode_known_words(tokenizer, sample_vocab):
    text = "Take the Falcon to Tatooine."
    expected_ids = [
        sample_vocab["Take"],
        sample_vocab["the"],
        sample_vocab["Falcon"],
        sample_vocab["to"],
        sample_vocab["Tatooine"],
        sample_vocab["."],
    ]
    assert tokenizer.encode(text) == expected_ids


def test_encode_unknown_tokens(tokenizer, sample_vocab):
    text = "Drive the Speeder."
    unk_id = sample_vocab["<|unk|>"]
    expected_ids = [
        unk_id,  # "Drive" is OOV
        sample_vocab["the"],
        unk_id,  # "Speeder" is OOV
        sample_vocab["."],
    ]
    assert tokenizer.encode(text) == expected_ids


def test_encode_empty_string(tokenizer):
    assert tokenizer.encode("") == []
    assert tokenizer.encode("    ") == []


def test_decode_reconstructs_text(tokenizer):
    text = "Take the Falcon back to Tatooine."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text


def test_decode_punctuation_spacing(tokenizer, sample_vocab):
    # Tests that regex strips spaces before commas, periods, etc.
    ids = [
        sample_vocab["Take"],
        sample_vocab[","],
        sample_vocab["Falcon"],
        sample_vocab["!"],
    ]
    decoded = tokenizer.decode(ids)
    assert decoded == "Take, Falcon!"


def test_roundtrip_with_unknowns(tokenizer):
    text = "Fly to Coruscant."
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == "<|unk|> to <|unk|>."


# --- Mocking Dataset Loading (Assignment Requirement) ---


def test_dataset_loading_and_vocab_build(mock_dataset):
    mock_file_content = json.dumps(mock_dataset)

    with (
        patch("builtins.open", mock_open(read_data=mock_file_content)),
        open("SW_EpisodeIV_VI.json", "r") as f,
    ):
        data = json.load(f)

    lines = [item["Line"] for item in data]
    preprocessed = process_text(" ".join(lines))
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: i for i, token in enumerate(all_tokens)}

    tok = SimpleTokenizer(vocab)
    encoded = tok.encode("Take the Falcon.")
    assert len(encoded) == 4
    assert tok.decode(encoded) == "Take the Falcon."
    mock_file_content = json.dumps(mock_dataset)

    with (
        patch("builtins.open", mock_open(read_data=mock_file_content)),
        open("SW_EpisodeIV_VI.json", "r") as f,
    ):
        data = json.load(f)

    lines = [item["Line"] for item in data]
    preprocessed = process_text(" ".join(lines))
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: i for i, token in enumerate(all_tokens)}

    tok = SimpleTokenizer(vocab)
    encoded = tok.encode("Take the Falcon.")
    assert len(encoded) == 4
    assert tok.decode(encoded) == "Take the Falcon."


# Tests for SentencePieceTokenizer
def test_initialization(untrained_sp_tokenizer):
    assert untrained_sp_tokenizer.str_to_int == {}
    assert untrained_sp_tokenizer.int_to_str == {}
    assert untrained_sp_tokenizer.merges == []


def test_process_text(untrained_sp_tokenizer):
    result = untrained_sp_tokenizer._process_text("hello world")

    assert result == [
        ("▁", "h", "e", "l", "l", "o"),
        ("▁", "w", "o", "r", "l", "d"),
    ]


def test_process_text_multiple_spaces(untrained_sp_tokenizer):
    result = untrained_sp_tokenizer._process_text("hello   world")

    assert result == [
        ("▁", "h", "e", "l", "l", "o"),
        ("▁", "w", "o", "r", "l", "d"),
    ]


def test_add_token(untrained_sp_tokenizer):
    untrained_sp_tokenizer._add_token("hello")

    assert untrained_sp_tokenizer.str_to_int["hello"] == 0
    assert untrained_sp_tokenizer.int_to_str[0] == "hello"


def test_add_token_does_not_duplicate(untrained_sp_tokenizer):
    untrained_sp_tokenizer._add_token("hello")
    untrained_sp_tokenizer._add_token("hello")

    assert len(untrained_sp_tokenizer.str_to_int) == 1
    assert len(untrained_sp_tokenizer.int_to_str) == 1
    assert untrained_sp_tokenizer.str_to_int["hello"] == 0


def test_get_pair_counts(untrained_sp_tokenizer):
    word_freqs = {
        ("▁", "a", "b"): 2,
        ("▁", "a", "c"): 1,
    }

    counts = untrained_sp_tokenizer._get_pair_counts(word_freqs)

    assert counts[("▁", "a")] == 3
    assert counts[("a", "b")] == 2
    assert counts[("a", "c")] == 1


def test_get_pair_counts_empty(untrained_sp_tokenizer):
    counts = untrained_sp_tokenizer._get_pair_counts({})

    assert not counts


def test_find_merge_candidate(untrained_sp_tokenizer):
    word_freqs = {
        ("▁", "a", "b"): 5,
        ("▁", "a", "c"): 2,
    }

    candidate = untrained_sp_tokenizer._find_merge_candidate(word_freqs)

    assert candidate == ("▁", "a")


def test_find_merge_candidate_no_pairs(untrained_sp_tokenizer):
    word_freqs = {
        ("a",): 10,
        ("b",): 5,
    }

    candidate = untrained_sp_tokenizer._find_merge_candidate(word_freqs)

    assert candidate is None


def test_merge_word(untrained_sp_tokenizer):
    word = ("▁", "h", "e", "l", "l", "o")

    result = untrained_sp_tokenizer._merge_word(
        word,
        ("l", "l"),
    )

    assert result == ("▁", "h", "e", "ll", "o")


def test_merge_word_multiple_occurrences(untrained_sp_tokenizer):
    word = ("a", "b", "a", "b")

    result = untrained_sp_tokenizer._merge_word(
        word,
        ("a", "b"),
    )

    assert result == ("ab", "ab")


def test_merge_word_no_match(untrained_sp_tokenizer):
    word = ("▁", "h", "i")

    result = untrained_sp_tokenizer._merge_word(
        word,
        ("x", "y"),
    )

    assert result == word


def test_train_adds_special_tokens(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello world", num_merges=0)

    assert "<|endoftext|>" in untrained_sp_tokenizer.str_to_int
    assert "<|unk|>" in untrained_sp_tokenizer.str_to_int

    endoftext_id = untrained_sp_tokenizer.str_to_int["<|endoftext|>"]
    unk_id = untrained_sp_tokenizer.str_to_int["<|unk|>"]

    assert untrained_sp_tokenizer.int_to_str[endoftext_id] == "<|endoftext|>"
    assert untrained_sp_tokenizer.int_to_str[unk_id] == "<|unk|>"


def test_train_adds_character_tokens(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello", num_merges=0)

    expected_tokens = {"▁", "h", "e", "l", "o"}

    assert expected_tokens.issubset(set(untrained_sp_tokenizer.str_to_int.keys()))


def test_train_zero_merges(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello world", num_merges=0)

    assert untrained_sp_tokenizer.merges == []


def test_train_creates_merges(sp_tokenizer):
    assert len(sp_tokenizer.merges) > 0


def test_train_respects_num_merges(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello world", num_merges=2)

    assert len(untrained_sp_tokenizer.merges) == 2


def test_train_creates_tokens_for_merges(sp_tokenizer):
    for first, second in sp_tokenizer.merges:
        merged_token = first + second

        assert merged_token in sp_tokenizer.str_to_int


def test_encode_known_text(sp_tokenizer):
    ids = sp_tokenizer.encode("hello world")

    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(token_id, int) for token_id in ids)


def test_encode_returns_valid_ids(sp_tokenizer):
    ids = sp_tokenizer.encode("hello world")

    for token_id in ids:
        assert token_id in sp_tokenizer.int_to_str


def test_encode_unknown_character(sp_tokenizer):
    # "#" was not present in the training data.
    ids = sp_tokenizer.encode("hello#")

    unk_id = sp_tokenizer.str_to_int["<|unk|>"]

    assert unk_id in ids


def test_decode_known_ids(sp_tokenizer):
    ids = sp_tokenizer.encode("hello world")

    decoded = sp_tokenizer.decode(ids)

    assert decoded == "hello world"


def test_decode_unknown_id(sp_tokenizer):
    decoded = sp_tokenizer.decode([999999])

    assert decoded == "<|unk|>"


def test_decode_empty_list(sp_tokenizer):
    assert sp_tokenizer.decode([]) == ""


def test_decode_sentencepiece_boundary(sp_tokenizer):
    boundary_id = sp_tokenizer.str_to_int["▁"]

    decoded = sp_tokenizer.decode([boundary_id])

    assert decoded == ""


def test_round_trip(sp_tokenizer):
    text = "hello world"

    encoded = sp_tokenizer.encode(text)
    decoded = sp_tokenizer.decode(encoded)

    assert decoded == text


def test_round_trip_repeated_words(sp_tokenizer):
    text = "hello hello world world"

    encoded = sp_tokenizer.encode(text)
    decoded = sp_tokenizer.decode(encoded)

    assert decoded == text


def test_encode_applies_merges(sp_tokenizer):
    ids = sp_tokenizer.encode("hello")

    # Without merges, "hello" would be represented by the
    # boundary marker plus five characters.
    assert len(ids) < 6


def test_retraining_resets_vocabulary(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello", num_merges=0)

    assert "h" in untrained_sp_tokenizer.str_to_int
    assert "w" not in untrained_sp_tokenizer.str_to_int

    untrained_sp_tokenizer.train("world", num_merges=0)

    assert "w" in untrained_sp_tokenizer.str_to_int
    assert "h" not in untrained_sp_tokenizer.str_to_int


def test_retraining_resets_reverse_vocabulary(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello", num_merges=0)

    first_vocab = dict(untrained_sp_tokenizer.int_to_str)

    untrained_sp_tokenizer.train("world", num_merges=0)

    second_vocab = dict(untrained_sp_tokenizer.int_to_str)

    assert first_vocab != second_vocab

    # The first token should once again be the word boundary marker.
    assert second_vocab[0] == "▁"


def test_retraining_resets_merges(untrained_sp_tokenizer):
    untrained_sp_tokenizer.train("hello hello", num_merges=5)

    assert len(untrained_sp_tokenizer.merges) > 0

    untrained_sp_tokenizer.train("a b c", num_merges=5)

    print(untrained_sp_tokenizer.merges)

    assert untrained_sp_tokenizer.merges == [
        ("\u2581", "a"),
        ("\u2581", "b"),
        ("\u2581", "c"),
    ]
