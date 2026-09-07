import pytest

from src.tokenizer import SimpleTokenizer, build_vocab


@pytest.fixture
def tokenizer():
    """A small, predictable vocabulary for tokenizer behavior tests."""
    return SimpleTokenizer(
        {
            "Hello": 0,
            "world": 1,
            ",": 2,
            "!": 3,
            "<|unk|>": 4,
        }
    )


@pytest.fixture
def sample_data():
    return [
        {"Line": "Hello, world!"},
        {"Line": "Hello galaxy"},
    ]


def test_build_vocab(sample_data):
    vocab = build_vocab(sample_data)

    assert "Hello" in vocab
    assert "," in vocab
    assert "world" in vocab
    assert "!" in vocab
    assert "galaxy" in vocab
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab


def test_encode_known_tokens(tokenizer):
    assert tokenizer.encode("Hello world") == [0, 1]


def test_encode_replaces_unknown_tokens(tokenizer):
    assert tokenizer.encode("Hello galaxy!") == [0, 4, 3]


def test_encode_separates_punctuation(tokenizer):
    assert tokenizer.encode("Hello, world!") == [0, 2, 1, 3]


def test_decode_reconstructs_text_and_attaches_punctuation(tokenizer):
    assert tokenizer.decode([0, 2, 1, 3]) == "Hello, world!"
