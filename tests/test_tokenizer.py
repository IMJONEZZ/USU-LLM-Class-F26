import pytest

from src.tokenizer import SimpleTokenizer


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


def test_encode_known_tokens(tokenizer):
    assert tokenizer.encode("Hello world") == [0, 1]


def test_encode_replaces_unknown_tokens(tokenizer):
    assert tokenizer.encode("Hello galaxy!") == [0, 4, 3]


def test_encode_separates_punctuation(tokenizer):
    assert tokenizer.encode("Hello, world!") == [0, 2, 1, 3]


def test_decode_reconstructs_text_and_attaches_punctuation(tokenizer):
    assert tokenizer.decode([0, 2, 1, 3]) == "Hello, world!"
