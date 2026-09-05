import json
from unittest.mock import mock_open, patch

import pytest

from src.tokenizer import SimpleTokenizer, process_text

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
