import pytest

from src.tokenizer import SimpleTokenizer


@pytest.fixture
def tokenizer():
    vocabulary = {
        "Hello": 0,
        ",": 1,
        "world": 2,
        "!": 3,
        "<|unk|>": 4,
    }
    return SimpleTokenizer(vocabulary)


def test_encode_known_words_and_punctuation(tokenizer):
    assert tokenizer.encode("Hello, world!") == [0, 1, 2, 3]


def test_encode_unknown_words_as_unknown_token(tokenizer):
    assert tokenizer.encode("Goodbye") == [4]


def test_encode_empty_text(tokenizer):
    assert tokenizer.encode("") == []


def test_decode_ids_and_punctuation(tokenizer):
    assert tokenizer.decode([0, 1, 2, 3]) == "Hello, world!"


def test_encode_decode_round_trip(tokenizer):
    text = "Hello, world!"
    assert tokenizer.decode(tokenizer.encode(text)) == text
