import pytest

from src.tokenizer import SimpleTokenizer, build_vocab, load_dialogue


@pytest.fixture
def sample_vocab():
    """A small, hand-built vocabulary for testing encode/decode logic."""
    tokens = [
        "Hello",
        "world",
        "this",
        "is",
        "a",
        "test",
        ",",
        ".",
        "!",
        "?",
        "<|endoftext|>",
        "<|unk|>",
    ]
    return {token: i for i, token in enumerate(tokens)}


@pytest.fixture
def tokenizer(sample_vocab):
    """A SimpleTokenizer built from the sample vocab, ready to use in tests."""
    return SimpleTokenizer(sample_vocab)


# ---------------------------------------------------------------------------
# ENCODE TESTS
# ---------------------------------------------------------------------------


def test_encode_returns_list_of_ints(tokenizer):
    """encode() should return a list of integers, not strings."""
    ids = tokenizer.encode("Hello world")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_encode_known_tokens(tokenizer, sample_vocab):
    """Encoding a known sentence should map each word to its correct ID."""
    ids = tokenizer.encode("Hello world")
    expected = [sample_vocab["Hello"], sample_vocab["world"]]
    assert ids == expected


def test_encode_handles_punctuation(tokenizer, sample_vocab):
    """Punctuation marks should be split out and tokenized individually."""
    ids = tokenizer.encode("Hello, world!")
    expected = [
        sample_vocab["Hello"],
        sample_vocab[","],
        sample_vocab["world"],
        sample_vocab["!"],
    ]
    assert ids == expected


def test_encode_unknown_word_maps_to_unk(tokenizer, sample_vocab):
    """A word not in the vocab should be replaced with <|unk|>, not crash."""
    ids = tokenizer.encode("Goodbye world")
    assert ids[0] == sample_vocab["<|unk|>"]
    assert ids[1] == sample_vocab["world"]


def test_encode_empty_string_returns_empty_list(tokenizer):
    """Encoding an empty string should return an empty list, not error."""
    ids = tokenizer.encode("")
    assert ids == []


def test_encode_multiple_unknown_words(tokenizer, sample_vocab):
    """Multiple out-of-vocab words should each become <|unk|>."""
    ids = tokenizer.encode("Foo bar baz")
    assert all(i == sample_vocab["<|unk|>"] for i in ids)


# ---------------------------------------------------------------------------
# DECODE TESTS
# ---------------------------------------------------------------------------


def test_decode_returns_string(tokenizer):
    """decode() should return a single string."""
    ids = tokenizer.encode("Hello world")
    result = tokenizer.decode(ids)
    assert isinstance(result, str)


def test_decode_known_ids(tokenizer, sample_vocab):
    """Decoding known IDs should reconstruct the original words in order."""
    ids = [sample_vocab["Hello"], sample_vocab["world"]]
    result = tokenizer.decode(ids)
    assert result == "Hello world"


def test_decode_removes_space_before_punctuation(tokenizer, sample_vocab):
    """Punctuation should not have a leading space after decoding."""
    ids = [sample_vocab["Hello"], sample_vocab[","], sample_vocab["world"]]
    result = tokenizer.decode(ids)
    assert result == "Hello, world"


# ---------------------------------------------------------------------------
# ROUND-TRIP TESTS
# ---------------------------------------------------------------------------


def test_round_trip_simple_sentence(tokenizer):
    """Encoding then decoding a simple sentence should return the same text."""
    text = "Hello world"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


def test_round_trip_with_punctuation(tokenizer):
    """Round-tripping text with punctuation should preserve it correctly."""
    text = "Hello, world!"
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    assert decoded == text


# ---------------------------------------------------------------------------
# VOCAB / MAPPING TESTS
# ---------------------------------------------------------------------------


def test_str_to_int_and_int_to_str_are_inverses(tokenizer, sample_vocab):
    """int_to_str should be the exact reverse mapping of str_to_int."""
    for token, idx in sample_vocab.items():
        assert tokenizer.int_to_str[idx] == token


def test_tokenizer_vocab_size_matches_input(tokenizer, sample_vocab):
    """The tokenizer's internal vocab should have the same size as the input."""
    assert len(tokenizer.str_to_int) == len(sample_vocab)


# ---------------------------------------------------------------------------
# build_vocab() TESTS
# ---------------------------------------------------------------------------


def test_build_vocab_contains_special_tokens():
    """build_vocab should always add <|endoftext|> and <|unk|> to the vocab."""
    vocab = build_vocab("hello world")
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab


def test_build_vocab_contains_expected_words():
    """build_vocab should include every unique word from the input text."""
    vocab = build_vocab("hello world hello")
    assert "hello" in vocab
    assert "world" in vocab
    # "hello" appears twice in the text but should only be one vocab entry
    assert len(vocab) == 2 + 2  # "hello", "world" + the 2 special tokens


def test_build_vocab_ids_are_unique():
    """Every token in the vocab should map to a unique integer ID."""
    vocab = build_vocab("the quick brown fox jumps over the lazy dog")
    ids = list(vocab.values())
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# load_dialogue() TEST
# ---------------------------------------------------------------------------


def test_load_dialogue_extracts_lines(tmp_path):
    """load_dialogue should read a JSON file and join all 'Line' fields."""
    fake_data = [
        {"Character": "LUKE", "Line": "Hello there."},
        {"Character": "LEIA", "Line": "General Kenobi."},
    ]
    fake_file = tmp_path / "fake_dataset.json"
    fake_file.write_text(__import__("json").dumps(fake_data), encoding="utf-8")

    result = load_dialogue(str(fake_file))

    assert "Hello there." in result
    assert "General Kenobi." in result
