"""Offline tests for token windows and final-position prediction batches."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def dataloader_module(monkeypatch):
    # The exported notebook downloads data and runs examples during import.
    datasets = ModuleType("datasets")
    train_split = MagicMock()
    train_split.__getitem__.return_value = {"Line": ["example"]}
    datasets.load_dataset = MagicMock(return_value={"train": train_split})
    tokenizer = MagicMock(pad_token_id=99, eos_token_id=98)
    tokenizer.encode.return_value = list(range(10))
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    path = Path(__file__).resolve().parents[1] / "src" / "dataloader.py"
    spec = importlib.util.spec_from_file_location("dataloader_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def make_dataset(dataloader_module):
    def make(tokens, *, max_length=3, step=2, pad=True, pad_id=99, eos_id=98):
        tokenizer = MagicMock(pad_token_id=pad_id, eos_token_id=eos_id)
        tokenizer.encode.return_value = tokens
        return dataloader_module.TextDataset(
            "example", tokenizer, max_length=max_length, step=step, pad=pad
        )

    return make


@pytest.mark.parametrize(
    ("tokens", "pad", "expected"),
    [
        ([], True, []),
        ([], False, []),
        ([1], True, [[99, 99, 99, 1]]),
        ([1, 2, 3], False, []),
        ([1, 2, 3], True, [[99, 1, 2, 3]]),
        ([1, 2, 3, 4], True, [[1, 2, 3, 4]]),
        ([1, 2, 3, 4], False, [[1, 2, 3, 4]]),
        ([1, 2, 3, 4, 5], True, [[1, 2, 3, 4], [99, 3, 4, 5]]),
        ([1, 2, 3, 4, 5], False, [[1, 2, 3, 4]]),
        ([1, 2, 3, 4, 5, 6], True, [[1, 2, 3, 4], [3, 4, 5, 6]]),
    ],
)
def test_chunks(make_dataset, tokens, pad, expected):
    original = tokens.copy()
    dataset = make_dataset(tokens, pad=pad)
    assert dataset.chunks == expected
    assert len(dataset) == len(expected)
    assert tokens == original
    dataset.tokenizer.encode.assert_called_once_with(
        "example", add_special_tokens=False
    )


def test_only_one_padded_window_with_small_step(make_dataset):
    dataset = make_dataset([1, 2, 3, 4, 5], max_length=4, step=1)
    assert dataset.chunks == [[1, 2, 3, 4, 5]]
    dataset = make_dataset([1, 2, 3], max_length=4, step=1)
    assert dataset.chunks == [[99, 99, 1, 2, 3]]


@pytest.mark.parametrize(("pad_id", "expected"), [(None, 98), (0, 0), (99, 99)])
def test_padding_token_fallback(make_dataset, pad_id, expected):
    dataset = make_dataset([1, 2], pad_id=pad_id)
    assert dataset.chunks == [[expected, expected, 1, 2]]


@pytest.mark.parametrize(
    ("tokens", "expected_inputs", "expected_target"),
    [
        ([1, 2, 3, 4], [1, 2, 3], 4),
        ([1, 2], [99, 99, 1], 2),
        ([1], [99, 99, 99], 1),
    ],
)
def test_final_position_target(make_dataset, tokens, expected_inputs, expected_target):
    dataset = make_dataset(tokens)
    inputs, targets = dataset[0]
    assert inputs.dtype == targets.dtype == torch.long
    assert inputs.shape == (3,)
    assert targets.shape == ()
    assert inputs.tolist() == expected_inputs
    assert targets.item() == expected_target


def test_minimum_sequence_length(make_dataset):
    dataset = make_dataset([1, 2, 3], max_length=1, step=1)
    assert dataset.chunks == [[1, 2], [2, 3]]


def test_index_boundaries(make_dataset):
    dataset = make_dataset([1, 2, 3, 4, 5])
    assert torch.equal(dataset[-1][0], dataset[1][0])
    for index in (2, -3):
        with pytest.raises(IndexError):
            dataset[index]
    with pytest.raises(IndexError):
        make_dataset([])[0]


def test_zero_step_rejected(make_dataset):
    with pytest.raises(ValueError):
        make_dataset([1, 2], step=0)


def test_dataloader_forwards_options(dataloader_module, monkeypatch):
    loader = MagicMock()
    monkeypatch.setattr(dataloader_module, "DataLoader", loader)
    dataset = object()
    result = dataloader_module.create_dataloader(
        dataset, batch_size=2, shuffle=False, num_workers=1
    )
    loader.assert_called_once_with(dataset, batch_size=2, shuffle=False, num_workers=1)
    assert result is loader.return_value


def test_batches_keep_partial_final_batch(dataloader_module, make_dataset):
    dataset = make_dataset(list(range(8)))
    batches = list(
        dataloader_module.create_dataloader(dataset, batch_size=2, shuffle=False)
    )
    assert [inputs.shape[0] for inputs, _ in batches] == [2, 1]
    assert batches[0][0].tolist() == [[0, 1, 2], [2, 3, 4]]
    assert batches[1][0].tolist() == [[4, 5, 6]]
    assert batches[0][1].shape == (2,)
    assert batches[0][1].tolist() == [3, 5]
    assert batches[1][1].shape == (1,)
    assert batches[1][1].tolist() == [7]


def test_empty_dataloader_without_shuffle(dataloader_module, make_dataset):
    loader = dataloader_module.create_dataloader(make_dataset([]), shuffle=False)
    assert list(loader) == []
