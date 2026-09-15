import pytest
import torch

from src.TextDataset import TextDataset


class FakeTokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.calls = []

    def encode(self, text, allowed_special=None):
        self.calls.append(
            {
                "text": text,
                "allowed_special": allowed_special,
            }
        )
        return self.token_ids


def test_tokenizer_is_called_correctly():
    tokenizer = FakeTokenizer([10, 20, 30, 40, 50])

    TextDataset(
        txt="example text",
        tokenizer=tokenizer,
        max_length=3,
        stride=1,
    )

    assert tokenizer.calls == [
        {
            "text": "example text",
            "allowed_special": {"<|endoftext|>"},
        }
    ]


def test_dataset_length():
    tokenizer = FakeTokenizer([0, 1, 2, 3, 4, 5])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=3,
        stride=1,
    )

    assert len(dataset) == 3


def test_input_and_target_are_shifted():
    tokenizer = FakeTokenizer([10, 20, 30, 40, 50])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=3,
        stride=1,
    )

    inputs, targets = dataset[0]

    assert torch.equal(inputs, torch.tensor([10, 20, 30]))
    assert torch.equal(targets, torch.tensor([20, 30, 40]))


def test_stride_controls_window_start():
    tokenizer = FakeTokenizer([0, 1, 2, 3, 4, 5, 6])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=3,
        stride=2,
    )

    first_inputs, first_targets = dataset[0]
    second_inputs, second_targets = dataset[1]

    assert torch.equal(first_inputs, torch.tensor([0, 1, 2]))
    assert torch.equal(first_targets, torch.tensor([1, 2, 3]))

    assert torch.equal(second_inputs, torch.tensor([2, 3, 4]))
    assert torch.equal(second_targets, torch.tensor([3, 4, 5]))


def test_samples_are_tensors():
    tokenizer = FakeTokenizer([0, 1, 2, 3])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=2,
        stride=1,
    )

    inputs, targets = dataset[0]

    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.dtype == torch.int64
    assert targets.dtype == torch.int64


def test_text_too_short_produces_empty_dataset():
    tokenizer = FakeTokenizer([0, 1, 2])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=3,
        stride=1,
    )

    assert len(dataset) == 0


def test_invalid_index_raises_index_error():
    tokenizer = FakeTokenizer([0, 1, 2, 3])

    dataset = TextDataset(
        txt="unused",
        tokenizer=tokenizer,
        max_length=2,
        stride=1,
    )

    with pytest.raises(IndexError):
        dataset[100]
