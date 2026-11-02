"""Tests for the sliding-window data loader.

Most of these use ``CountingTokenizer`` rather than a real BPE vocabulary. It
returns 0, 1, 2, ... one ID per character, which means the token at stream
position *i* is literally the integer *i*. That turns assertions that would
otherwise be vague into exact ones: a window starting at 5 with length 4 must
be ``[5, 6, 7, 8]``, and "do these two datasets share any token?" becomes a set
intersection on values.

It also keeps the suite fast and stops these tests from failing when the
tokenizer changes, since what is under test here is the windowing and batching,
not BPE. Two tests at the end use the real tokenizer to confirm the pieces fit
together.
"""

import json

import pytest
import torch

from src.bpe_tokenizer import BPETokenizer
from src.dataloader import (
    DEFAULT_MAX_LENGTH,
    SlidingWindowDataset,
    build_dataloaders,
    create_dataloader,
    join_lines,
    load_lines,
    main,
    train_val_datasets,
)


class CountingTokenizer:
    """Encodes text as 0, 1, 2, ... so token value == stream position."""

    def encode(self, text):
        return list(range(len(text)))


@pytest.fixture
def tokenizer():
    return CountingTokenizer()


@pytest.fixture
def dataset():
    """1,000 tokens, windows of 10, disjoint."""
    return SlidingWindowDataset(list(range(1000)), max_length=10)


@pytest.fixture
def dataset_file(tmp_path, monkeypatch):
    """A temp cwd holding a dataset file big enough for the real defaults.

    ``main`` uses ``max_length=256`` and pulls a batch, so the corpus has to
    survive the 90/10 split and still produce at least one full window.
    """
    lines = [
        f"Character number {i} says something about the Force and the Empire."
        for i in range(400)
    ]
    records = [
        {"Character": f"SPEAKER{i}", "Line": line} for i, line in enumerate(lines)
    ]
    (tmp_path / "SW_EpisodeIV_VI.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- the shift, checked on values rather than shapes -------------------------


def test_target_is_input_shifted_left_by_exactly_one(dataset):
    """A shape-only test would pass even if the shift were wrong or absent."""
    inputs, targets = dataset[0]
    assert inputs.tolist() == list(range(10))
    assert targets.tolist() == list(range(1, 11))


def test_shift_holds_for_every_window(dataset):
    for index in range(len(dataset)):
        inputs, targets = dataset[index]
        assert torch.equal(inputs[1:], targets[:-1])


def test_window_starts_advance_by_stride():
    data = SlidingWindowDataset(list(range(100)), max_length=10, stride=10)
    assert data[0][0][0].item() == 0
    assert data[1][0][0].item() == 10
    assert data[2][0][0].item() == 20


def test_items_are_long_tensors_of_the_right_shape(dataset):
    inputs, targets = dataset[0]
    assert inputs.dtype is torch.long
    assert targets.dtype is torch.long
    assert inputs.shape == (10,)
    assert targets.shape == (10,)


def test_getitem_returns_views_not_copies(dataset):
    """Slices should share storage with the one tensor built in __init__."""
    inputs, _ = dataset[0]
    assert (
        inputs.untyped_storage().data_ptr()
        == dataset.token_ids.untyped_storage().data_ptr()
    )


# --- chunk counts, stride, and the duplication overlap causes ----------------


@pytest.mark.parametrize(
    ("length", "max_length", "stride", "expected"),
    [
        (1000, 10, 10, 99),  # disjoint: last window needs a token to shift onto
        (1000, 10, 5, 198),  # half overlap: roughly twice as many windows
        (1000, 10, 1, 990),  # maximum overlap
        (100, 10, 10, 9),
    ],
)
def test_chunk_count_matches_length_and_stride(length, max_length, stride, expected):
    data = SlidingWindowDataset(list(range(length)), max_length, stride)
    assert len(data) == expected


def test_overlapping_stride_reuses_tokens():
    """Halving the stride roughly doubles how often each token is used."""
    disjoint = SlidingWindowDataset(list(range(1000)), max_length=10, stride=10)
    overlapping = SlidingWindowDataset(list(range(1000)), max_length=10, stride=5)
    assert len(overlapping) > len(disjoint)
    # Same underlying tokens, about twice the training examples.
    assert 1.9 < len(overlapping) / len(disjoint) < 2.1


def test_corpus_shorter_than_the_window_yields_nothing():
    """Better to produce no chunks than one short, silently padded one."""
    assert len(SlidingWindowDataset(list(range(10)), max_length=256)) == 0
    assert len(SlidingWindowDataset([], max_length=4)) == 0


def test_corpus_exactly_one_token_longer_than_the_window():
    """The boundary case: max_length + 1 tokens is exactly one window."""
    assert len(SlidingWindowDataset(list(range(11)), max_length=10)) == 1
    assert len(SlidingWindowDataset(list(range(10)), max_length=10)) == 0


def test_stride_defaults_to_max_length(dataset):
    assert dataset.stride == dataset.max_length


@pytest.mark.parametrize("max_length", [0, -1])
def test_invalid_max_length_is_rejected(max_length):
    with pytest.raises(ValueError, match="max_length"):
        SlidingWindowDataset(list(range(100)), max_length=max_length)


@pytest.mark.parametrize("stride", [0, -3])
def test_invalid_stride_is_rejected(stride):
    with pytest.raises(ValueError, match="stride"):
        SlidingWindowDataset(list(range(100)), max_length=10, stride=stride)


# --- splitting before windowing, and the leak it prevents --------------------


def test_train_and_validation_share_no_tokens(tokenizer):
    """The point of splitting the stream before windowing it.

    Token values are stream positions here, so any shared value would be a
    token appearing on both sides of the split.
    """
    text = "x" * 1000
    train, validation = train_val_datasets(text, tokenizer, max_length=10)

    train_tokens = set()
    for index in range(len(train)):
        train_tokens.update(train[index][1].tolist())
    val_tokens = set()
    for index in range(len(validation)):
        val_tokens.update(validation[index][1].tolist())

    assert train_tokens and val_tokens
    assert train_tokens & val_tokens == set()


def test_no_leak_even_with_heavily_overlapping_windows(tokenizer):
    """Overlap is where window-then-split would leak badly."""
    text = "x" * 1000
    train, validation = train_val_datasets(text, tokenizer, max_length=10, stride=1)

    train_tokens = set()
    for index in range(len(train)):
        train_tokens.update(train[index][1].tolist())
    val_tokens = set()
    for index in range(len(validation)):
        val_tokens.update(validation[index][1].tolist())

    assert train_tokens & val_tokens == set()


def test_validation_tokens_all_come_after_the_cut(tokenizer):
    text = "x" * 1000
    train, validation = train_val_datasets(text, tokenizer, max_length=10)
    cut = 900

    assert max(train[len(train) - 1][1].tolist()) < cut
    assert min(validation[0][0].tolist()) >= cut


def test_val_fraction_controls_the_split(tokenizer):
    text = "x" * 1000
    train, validation = train_val_datasets(
        text, tokenizer, max_length=10, val_fraction=0.5
    )
    assert len(train.token_ids) == 500
    assert len(validation.token_ids) == 500


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.5])
def test_invalid_val_fraction_is_rejected(tokenizer, val_fraction):
    with pytest.raises(ValueError, match="val_fraction"):
        train_val_datasets("x" * 100, tokenizer, val_fraction=val_fraction)


# --- batching ----------------------------------------------------------------


def test_batches_have_the_expected_shape(dataset):
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    inputs, targets = next(iter(loader))
    assert inputs.shape == (4, 10)
    assert targets.shape == (4, 10)


def test_batching_preserves_the_shift(dataset):
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    inputs, targets = next(iter(loader))
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_drop_last_discards_the_short_final_batch(dataset):
    """99 chunks at batch_size 4 is 24 full batches with 3 left over."""
    assert len(dataset) == 99
    kept = create_dataloader(dataset, batch_size=4, drop_last=False)
    dropped = create_dataloader(dataset, batch_size=4, drop_last=True)
    assert len(kept) == 25
    assert len(dropped) == 24


def test_final_batch_is_short_when_not_dropped(dataset):
    loader = create_dataloader(dataset, batch_size=4, shuffle=False, drop_last=False)
    last_inputs, _ = list(loader)[-1]
    assert last_inputs.shape[0] == 3


def test_every_chunk_appears_exactly_once_per_epoch(dataset):
    loader = create_dataloader(dataset, batch_size=4, shuffle=True, seed=0)
    seen = []
    for inputs, _ in loader:
        seen.extend(int(row[0]) for row in inputs)
    assert sorted(seen) == [start for start in dataset.starts]


def test_num_workers_defaults_to_zero(dataset):
    assert create_dataloader(dataset).num_workers == 0


# --- reproducibility ---------------------------------------------------------


def _batch_order(dataset, seed):
    loader = create_dataloader(dataset, batch_size=4, shuffle=True, seed=seed)
    return [int(inputs[0, 0]) for inputs, _ in loader]


def test_same_seed_gives_the_same_batch_order(dataset):
    assert _batch_order(dataset, 0) == _batch_order(dataset, 0)


def test_different_seeds_give_different_batch_orders(dataset):
    assert _batch_order(dataset, 0) != _batch_order(dataset, 1)


def test_shuffling_off_preserves_dataset_order(dataset):
    loader = create_dataloader(dataset, batch_size=4, shuffle=False)
    first_column = [int(inputs[0, 0]) for inputs, _ in loader]
    assert first_column == sorted(first_column)


# --- reading the dataset file ------------------------------------------------


def test_load_lines_returns_only_dialogue(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"Character": "THREEPIO", "Line": "Did you hear that?"},
                {"Character": "LUKE", "Line": "I'm going to Tosche Station."},
            ]
        ),
        encoding="utf-8",
    )
    assert load_lines(path) == ["Did you hear that?", "I'm going to Tosche Station."]


def test_load_lines_skips_records_without_usable_text(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"Character": "LUKE", "Line": "Keep this one."},
                {"Character": "NOBODY"},
                {"Character": "EMPTY", "Line": ""},
                "not a record at all",
            ]
        ),
        encoding="utf-8",
    )
    assert load_lines(path) == ["Keep this one."]


def test_join_lines_inserts_the_separator():
    assert join_lines(["a", "b", "c"], separator="|") == "a|b|c"


def test_join_lines_uses_the_end_of_text_token_by_default():
    assert "<|endoftext|>" in join_lines(["a", "b"])


# --- the pieces fitting together with the real tokenizer ---------------------


def test_from_text_uses_the_supplied_tokenizer(tokenizer):
    data = SlidingWindowDataset.from_text("x" * 100, tokenizer, max_length=10)
    assert len(data) == 9
    assert data[0][0].tolist() == list(range(10))


def test_works_with_the_real_bpe_tokenizer():
    """One end-to-end check that BPE output feeds the windowing correctly."""
    text = join_lines(["Luke, use the Force."] * 60)
    bpe = BPETokenizer().train(text, vocab_size=300)
    data = SlidingWindowDataset.from_text(text, bpe, max_length=16)

    assert len(data) > 0
    inputs, targets = data[0]
    assert inputs.shape == (16,)
    assert torch.equal(inputs[1:], targets[:-1])


def test_build_dataloaders_end_to_end(dataset_file, tokenizer):
    train_loader, val_loader = build_dataloaders(
        tokenizer=tokenizer, max_length=32, batch_size=4, seed=0
    )
    inputs, targets = next(iter(train_loader))

    assert inputs.shape == (4, 32)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert len(val_loader.dataset) > 0


def test_build_dataloaders_trains_a_tokenizer_when_none_is_given(dataset_file):
    train_loader, _ = build_dataloaders(
        max_length=16, batch_size=2, vocab_size=300, seed=0
    )
    assert len(train_loader.dataset) > 0


def test_validation_loader_is_not_shuffled(dataset_file, tokenizer):
    _, val_loader = build_dataloaders(
        tokenizer=tokenizer, max_length=32, batch_size=4, seed=0
    )
    first_column = [int(inputs[0, 0]) for inputs, _ in val_loader]
    assert first_column == sorted(first_column)


def test_main_reports_the_expected_fields(dataset_file, capsys):
    main()
    output = capsys.readouterr().out

    assert "dialogue lines      400" in output
    assert f"window size         {DEFAULT_MAX_LENGTH}" in output
    assert "target is shifted   True" in output
