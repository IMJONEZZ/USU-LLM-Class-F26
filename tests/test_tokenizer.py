import pytest

from src.tokenizer import SimpleTokenizer, build_vocab


@pytest.fixture
def vocab():
    tokens = ["Hello", "world", ",", ".", "<|endoftext|>", "<|unk|>"]
    return {token: i for i, token in enumerate(sorted(tokens))}


@pytest.fixture
def tokenizer(vocab):
    return SimpleTokenizer(vocab)


def test_build_vocab_includes_special_tokens():
    vocab = build_vocab(["Hello", "world"])
    assert vocab["Hello"] == 0
    assert vocab["world"] == 1
    assert vocab["<|endoftext|>"] == 2
    assert vocab["<|unk|>"] == 3


def test_encode_known_tokens(tokenizer, vocab):
    ids = tokenizer.encode("Hello, world.")
    assert ids == [vocab["Hello"], vocab[","], vocab["world"], vocab["."]]


def test_encode_unknown_word_maps_to_unk(tokenizer, vocab):
    ids = tokenizer.encode("Goodbye world.")
    assert ids[0] == vocab["<|unk|>"]


def test_decode_round_trip(tokenizer):
    text = "Hello, world."
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_vocab_contains_special_tokens(vocab):
    assert "<|endoftext|>" in vocab
    assert "<|unk|>" in vocab
