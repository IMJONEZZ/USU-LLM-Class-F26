import json
import runpy
from pathlib import Path
from unittest.mock import mock_open

import pytest


@pytest.fixture(scope="module")
def dialogue():
    """Small dataset with the same fields as the real Star Wars JSON."""
    return [
        {"Character": "LUKE", "Line": "Hello, world!"},
        {"Character": "LEIA", "Line": "Hello, café!"},
    ]


@pytest.fixture(scope="module")
def tokenizer_script(dialogue):
    """Run the unchanged script structure with a fake dataset file for CI."""
    path = Path(__file__).resolve().parents[1] / "src" / "tokenizer.py"
    fake_open = mock_open(read_data=json.dumps(dialogue, ensure_ascii=False))
    return runpy.run_path(str(path), init_globals={"open": fake_open})


@pytest.fixture
def tokenizer(tokenizer_script):
    # Use fixed, non-consecutive IDs so we can check exact expected results.
    vocab = {
        "Hello": 10,
        "world": 20,
        ",": 30,
        "!": 40,
        "<|endoftext|>": 50,
        "<|unk|>": 60,
    }
    return tokenizer_script["SimpleTokenizer"](vocab)


def test_loads_dialogue_and_splits_punctuation(tokenizer_script):
    assert tokenizer_script["text"] == "Hello, world!\nHello, café!"
    assert tokenizer_script["preprocessed"] == [
        "Hello",
        ",",
        "world",
        "!",
        "Hello",
        ",",
        "café",
        "!",
    ]


def test_builds_sorted_unique_vocabulary(tokenizer_script):
    assert tokenizer_script["vocab"] == {
        "!": 0,
        ",": 1,
        "Hello": 2,
        "café": 3,
        "world": 4,
        "<|endoftext|>": 5,
        "<|unk|>": 6,
    }


def test_creates_tokenizer_from_dataset(tokenizer_script):
    tokenizer = tokenizer_script["tokenizer"]
    ids = tokenizer.encode(tokenizer_script["text"])
    assert ids == [2, 1, 4, 0, 2, 1, 3, 0]
    assert tokenizer.decode(ids) == "Hello, world! Hello, café!"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello world", [10, 20]),
        ("world Hello world", [20, 10, 20]),
        ("Hello, world!", [10, 30, 20, 40]),
        ("Hello stranger", [10, 60]),
        ("hello", [60]),  # The skeleton is case-sensitive.
        ("<|endoftext|> <|unk|>", [50, 60]),
        ("  Hello\tworld\n", [10, 20]),
        ("", []),
        (" \t\n", []),
    ],
)
def test_encode(tokenizer, text, expected):
    assert tokenizer.encode(text) == expected


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([10, 20], "Hello world"),
        ([20, 10, 20], "world Hello world"),
        ([10, 30, 20, 40], "Hello, world!"),
        ([10, 60, 50], "Hello <|unk|> <|endoftext|>"),
        ([], ""),
    ],
)
def test_decode(tokenizer, ids, expected):
    assert tokenizer.decode(ids) == expected


def test_decode_unknown_id_raises_key_error(tokenizer):
    with pytest.raises(KeyError):
        tokenizer.decode([999])


def test_round_trip_normalizes_whitespace(tokenizer):
    assert tokenizer.decode(tokenizer.encode("  Hello,\tworld!\n")) == "Hello, world!"


@pytest.mark.parametrize("punctuation", list(",.:;?_!\"()'") + ["--"])
def test_encode_splits_each_supported_punctuation(tokenizer_script, punctuation):
    tokenizer = tokenizer_script["SimpleTokenizer"](
        {"Hello": 0, punctuation: 1, "world": 2, "<|unk|>": 3}
    )
    assert tokenizer.encode(f"Hello{punctuation}world") == [0, 1, 2]
