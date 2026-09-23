import json

import pytest

from src.tokenizer import (
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    SimpleTokenizer,
    build_vocab,
    load_dialogue,
    main,
    split_text,
)


@pytest.fixture
def sample_vocab() -> dict[str, int]:
    """Create a small case-preserving vocabulary for tests."""
    return build_vocab("Hello hello, world!")


@pytest.fixture
def tokenizer(sample_vocab: dict[str, int]) -> SimpleTokenizer:
    """Create a tokenizer using the sample vocabulary."""
    return SimpleTokenizer(sample_vocab)


def test_split_text_preserves_case_and_separates_punctuation():
    tokens = split_text("Hello hello, world!")

    assert tokens == ["Hello", "hello", ",", "world", "!"]


def test_build_vocab_contains_special_tokens(sample_vocab):
    assert END_OF_TEXT_TOKEN in sample_vocab
    assert UNKNOWN_TOKEN in sample_vocab


def test_build_vocab_preserves_case(sample_vocab):
    assert "Hello" in sample_vocab
    assert "hello" in sample_vocab
    assert sample_vocab["Hello"] != sample_vocab["hello"]


def test_encode_known_words_and_punctuation(tokenizer):
    encoded = tokenizer.encode("Hello, world!")

    expected = [
        tokenizer.str_to_int["Hello"],
        tokenizer.str_to_int[","],
        tokenizer.str_to_int["world"],
        tokenizer.str_to_int["!"],
    ]
    assert encoded == expected


def test_encode_unknown_word(tokenizer):
    encoded = tokenizer.encode("Vader")

    assert encoded == [tokenizer.str_to_int[UNKNOWN_TOKEN]]


def test_decode_encoded_text(tokenizer):
    original = "Hello, world!"
    encoded = tokenizer.encode(original)

    assert tokenizer.decode(encoded) == original


def test_load_dialogue_uses_only_line_field(tmp_path):
    dataset = [
        {"Character": "LUKE", "Line": "Hello there."},
        {"Character": "LEIA", "Line": "General."},
    ]
    dataset_path = tmp_path / "dialogue.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    dialogue = load_dialogue(dataset_path)

    assert dialogue == ["Hello there.", "General."]
    assert "LUKE" not in dialogue
    assert "LEIA" not in dialogue


def test_main_runs_with_dataset(tmp_path, monkeypatch, capsys):
    dataset = [{"Character": "THREEPIO", "Line": "We are ready!"}]
    dataset_path = tmp_path / "dialogue.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    monkeypatch.setattr("src.tokenizer.DATASET_PATH", dataset_path)

    main()

    output = capsys.readouterr().out
    assert "Vocabulary size:" in output
    assert "Original: We are ready!" in output
    assert "Decoded: We are ready!" in output
