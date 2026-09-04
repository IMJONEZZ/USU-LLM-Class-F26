import pytest

from src.tokenizer import SimpleTokenizer


@pytest.fixture
def sample_vocab():
    """Provides a small, deterministic vocabulary for testing."""
    tokens = [
        "!",
        ",",
        "--",
        ".",
        "Force",
        "Hello",
        "May",
        "the",
        "world",
        "<|endoftext|>",
        "<|unk|>",
    ]
    return {token: idx for idx, token in enumerate(tokens)}


@pytest.fixture
def tokenizer(sample_vocab):
    """Provides a ready-to-use SimpleTokenizer instance."""
    return SimpleTokenizer(sample_vocab)


@pytest.fixture
def sample_star_wars_dataset():
    """Simulates a small slice of Star Wars dialogue data without reading from disk."""
    return [
        {"Line": "May the Force be with you."},
        {"Line": "Hello, world!"},
    ]


def test_tokenizer_initialization(sample_vocab):
    """Verifies internal mapping dictionaries build correctly."""
    tok = SimpleTokenizer(sample_vocab)
    assert tok.str_to_int == sample_vocab
    assert tok.int_to_str[sample_vocab["Hello"]] == "Hello"
    assert len(tok.int_to_str) == len(sample_vocab)


def test_encode_known_tokens(tokenizer, sample_vocab):
    """Tests encoding when all words and punctuation are in the vocab."""
    text = "Hello, world!"
    expected_ids = [
        sample_vocab["Hello"],
        sample_vocab[","],
        sample_vocab["world"],
        sample_vocab["!"],
    ]
    assert tokenizer.encode(text) == expected_ids


def test_encode_unknown_tokens(tokenizer, sample_vocab):
    """Verifies out-of-vocabulary words map to the <|unk|> ID."""
    text = "Luke Skywalker"  # Neither word is in sample_vocab
    unk_id = sample_vocab["<|unk|>"]
    assert tokenizer.encode(text) == [unk_id, unk_id]


def test_encode_handles_extra_whitespace(tokenizer, sample_vocab):
    """Verifies that tabs, multiple spaces, and newlines don't generate empty tokens."""
    text = "  Hello \n\t  world !  "
    expected_ids = [
        sample_vocab["Hello"],
        sample_vocab["world"],
        sample_vocab["!"],
    ]
    assert tokenizer.encode(text) == expected_ids


def test_encode_special_punctuation(tokenizer, sample_vocab):
    """Verifies punctuation like double dashes are preserved as distinct tokens."""
    text = "Hello--world"
    expected_ids = [
        sample_vocab["Hello"],
        sample_vocab["--"],
        sample_vocab["world"],
    ]
    assert tokenizer.encode(text) == expected_ids


def test_decode_basic(tokenizer, sample_vocab):
    """Tests basic token-to-string decoding."""
    ids = [sample_vocab["May"], sample_vocab["the"], sample_vocab["Force"]]
    assert tokenizer.decode(ids) == "May the Force"


def test_decode_strips_punctuation_whitespace(tokenizer, sample_vocab):
    """Verifies regex cleans spaces preceding punctuation (e.g., 'Hello ,' -> 'Hello,')."""
    ids = [
        sample_vocab["Hello"],
        sample_vocab[","],
        sample_vocab["world"],
        sample_vocab["!"],
    ]
    # Without the regex cleanup it would be "Hello , world !"
    assert tokenizer.decode(ids) == "Hello, world!"


def test_round_trip_encoding_decoding(tokenizer):
    """Verifies text encoded and then decoded retains its original structure."""
    original = "Hello, world!"
    assert tokenizer.decode(tokenizer.encode(original)) == original


def test_encode_empty_string(tokenizer):
    """Edge case: encoding an empty string or pure whitespace returns an empty list."""
    assert tokenizer.encode("") == []
    assert tokenizer.encode("    \n\t  ") == []


def test_decode_empty_list(tokenizer):
    """Edge case: decoding an empty list returns an empty string."""
    assert tokenizer.decode([]) == ""
