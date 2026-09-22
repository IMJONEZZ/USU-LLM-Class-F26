"""
Tests for the TextDataset and DataLoader helpers in src/dataloader.py.

"""

import json

import pytest

from src.bpe_tokenizer import BPETokenizer, train_bpe
from src.dataloader import (
    SPECIAL_TOKEN,
    TextDataset,
    build_training_text,
    create_dataloader,
    encode_with_special_token,
)

# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def training_text():
    """Small repetitive text so BPE has clear, predictable merges to learn."""
    return "the cat sat on the mat the cat ran the dog sat on the mat"


@pytest.fixture
def tokenizer(training_text):
    """A BPETokenizer trained on the small sample text above."""
    merges, vocab = train_bpe(training_text, num_merges=20)
    return BPETokenizer(merges, vocab)


@pytest.fixture
def text_with_special_token(training_text):
    """Sample text containing the special end-of-text separator token."""
    return f"the cat sat {SPECIAL_TOKEN} the dog ran {SPECIAL_TOKEN} we are doomed"


# ---------------------------------------------------------------------------
# build_training_text() TESTS
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_build_training_text_joins_lines_with_separator(tmp_path):
    """Lines from the dataset should be joined together with the separator token."""
    fake_data = [
        {"Character": "LUKE", "Line": "Hello there."},
        {"Character": "LEIA", "Line": "General Kenobi."},
    ]
    fake_file = tmp_path / "fake_dataset.json"
    fake_file.write_text(json.dumps(fake_data), encoding="utf-8")

    result = build_training_text(data_path=str(fake_file))

    assert "Hello there." in result
    assert "General Kenobi." in result
    assert SPECIAL_TOKEN in result


def test_build_training_text_uses_custom_separator(tmp_path):
    """A custom separator should be used instead of the default when provided."""
    fake_data = [
        {"Character": "LUKE", "Line": "Hello there."},
        {"Character": "LEIA", "Line": "General Kenobi."},
    ]
    fake_file = tmp_path / "fake_dataset.json"
    fake_file.write_text(json.dumps(fake_data), encoding="utf-8")

    result = build_training_text(data_path=str(fake_file), separator="<SEP>")

    assert "<SEP>" in result
    assert SPECIAL_TOKEN not in result


# ---------------------------------------------------------------------------
# encode_with_special_token() TESTS
# ---------------------------------------------------------------------------


def test_encode_with_special_token_inserts_special_id(
    tokenizer, text_with_special_token
):
    """The special token's ID should appear in the encoded output."""
    ids = encode_with_special_token(tokenizer, text_with_special_token)
    special_id = tokenizer.str_to_int[SPECIAL_TOKEN]
    assert special_id in ids


def test_encode_with_special_token_correct_count(tokenizer, text_with_special_token):
    """The special token should appear exactly as many times as it does in the text."""
    ids = encode_with_special_token(tokenizer, text_with_special_token)
    special_id = tokenizer.str_to_int[SPECIAL_TOKEN]
    expected_count = text_with_special_token.count(SPECIAL_TOKEN)
    actual_count = ids.count(special_id)
    assert actual_count == expected_count


def test_encode_with_special_token_no_special_token_present(tokenizer):
    """Text without the special token should encode normally, with no special ID inserted."""
    ids = encode_with_special_token(tokenizer, "the cat sat")
    special_id = tokenizer.str_to_int[SPECIAL_TOKEN]
    assert special_id not in ids


def test_encode_with_special_token_does_not_split_special_token(tokenizer):
    """
    The special token should never be broken into subword pieces -- it should
    map to exactly one ID, not multiple IDs from character-level splitting.
    """
    ids = encode_with_special_token(tokenizer, SPECIAL_TOKEN)
    assert len(ids) == 1
    assert ids[0] == tokenizer.str_to_int[SPECIAL_TOKEN]


# ---------------------------------------------------------------------------
# TextDataset TESTS
# ---------------------------------------------------------------------------


def test_text_dataset_length_matches_expected_chunk_count(tokenizer, training_text):
    """
    The number of samples should match how many sliding-window chunks fit in
    the tokenized text, given max_length and stride.
    """
    max_length = 5
    stride = 2
    dataset = TextDataset(training_text, tokenizer, max_length, stride)

    token_ids = encode_with_special_token(tokenizer, training_text)
    expected_len = len(range(0, len(token_ids) - max_length, stride))

    assert len(dataset) == expected_len


def test_text_dataset_getitem_returns_input_target_pair(tokenizer, training_text):
    """Each item should be a tuple of (input, target) tensors."""
    dataset = TextDataset(training_text, tokenizer, max_length=5, stride=2)
    input_ids, target_ids = dataset[0]

    assert len(input_ids) == 5
    assert len(target_ids) == 5


def test_text_dataset_target_is_input_shifted_by_one(tokenizer, training_text):
    """
    The target sequence should be exactly the input sequence shifted forward
    by one token -- this is the core requirement for next-token-prediction
    training, so it's worth testing explicitly rather than just checking shape.
    """
    dataset = TextDataset(training_text, tokenizer, max_length=5, stride=2)
    input_ids, target_ids = dataset[0]

    # target[i] should equal input[i+1] for every position except the last
    for i in range(len(input_ids) - 1):
        assert target_ids[i].item() == input_ids[i + 1].item()


def test_text_dataset_multiple_chunks_differ(tokenizer, training_text):
    """With a stride smaller than max_length, consecutive chunks should overlap but not be identical."""
    dataset = TextDataset(training_text, tokenizer, max_length=5, stride=2)
    assert len(dataset) > 1

    first_input, _ = dataset[0]
    second_input, _ = dataset[1]
    assert not torch_tensors_equal(first_input, second_input)


def torch_tensors_equal(a, b):
    """Helper to compare two tensors for exact equality."""
    return a.shape == b.shape and bool((a == b).all())


# ---------------------------------------------------------------------------
# create_dataloader() TESTS
# ---------------------------------------------------------------------------


def test_create_dataloader_returns_correct_batch_shape(tokenizer, training_text):
    """The DataLoader should yield batches with the requested batch_size and max_length."""
    dataloader = create_dataloader(
        training_text,
        tokenizer,
        batch_size=2,
        max_length=5,
        stride=2,
        shuffle=False,
        drop_last=False,
    )
    inputs, targets = next(iter(dataloader))

    assert inputs.shape[1] == 5
    assert targets.shape[1] == 5
    assert inputs.shape[0] <= 2


def test_create_dataloader_drop_last_removes_incomplete_batch(tokenizer, training_text):
    """
    With drop_last=True, the DataLoader should not yield a final batch smaller
    than batch_size, even if the dataset size doesn't divide evenly.
    """
    dataloader = create_dataloader(
        training_text,
        tokenizer,
        batch_size=1000,  # deliberately larger than the dataset to force this case
        max_length=5,
        stride=2,
        shuffle=False,
        drop_last=True,
    )
    assert len(dataloader) == 0
