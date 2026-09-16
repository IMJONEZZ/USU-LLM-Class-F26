"""Check sequence construction separately from PyTorch batch assembly."""

from itertools import pairwise
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src.bpe_tokenizer import BPETokenizer, load_corpus
from src.dataloader import TextDataset, create_dataloader


@pytest.fixture
def tokenizer():
    # The initial byte vocabulary gives ASCII text predictable token IDs.
    return BPETokenizer()


@pytest.mark.parametrize(
    ("text", "max_length", "stride", "expected_inputs", "expected_targets"),
    [
        ("ABCDE", 4, None, ["ABCD"], ["BCDE"]),
        ("ABCDEF", 4, None, ["ABCD", "BCDE"], ["BCDE", "CDEF"]),
        ("ABCDEFGHI", 4, None, ["ABCD", "EFGH"], ["BCDE", "FGHI"]),
        (
            "ABCDEFGHIJ",
            4,
            None,
            ["ABCD", "EFGH", "FGHI"],
            ["BCDE", "FGHI", "GHIJ"],
        ),
        (
            "ABCDEFGHIJ",
            4,
            2,
            ["ABCD", "CDEF", "EFGH", "FGHI"],
            ["BCDE", "DEFG", "FGHI", "GHIJ"],
        ),
        ("ABCDEFG", 4, 2, ["ABCD", "CDEF"], ["BCDE", "DEFG"]),
        ("ABC", 1, None, ["A", "B"], ["B", "C"]),
        ("ABCDEF", 4, 1, ["ABCD", "BCDE"], ["BCDE", "CDEF"]),
    ],
)
def test_exact_windows_and_tail(
    tokenizer, text, max_length, stride, expected_inputs, expected_targets
):
    dataset = TextDataset(text, tokenizer, max_length, stride)

    assert len(dataset) == len(expected_inputs)
    for (inputs, targets), expected_input, expected_target in zip(
        dataset, expected_inputs, expected_targets, strict=True
    ):
        assert inputs.tolist() == list(expected_input.encode())
        assert targets.tolist() == list(expected_target.encode())
        assert inputs.dtype == targets.dtype == torch.long
        assert inputs.device.type == targets.device.type == "cpu"


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_every_next_token_transition_is_covered(tokenizer, stride):
    text = "ABCDEFGHIJKLMN"
    dataset = TextDataset(text, tokenizer, max_length=4, stride=stride)
    observed = {
        pair
        for inputs, targets in dataset
        for pair in zip(inputs.tolist(), targets.tolist(), strict=True)
    }
    source = list(text.encode())
    assert observed == set(pairwise(source))


@pytest.mark.parametrize("text", ["", "A", "ABCD"])
def test_insufficient_data_raises_helpful_error(tokenizer, text):
    with pytest.raises(ValueError, match="Need at least 5 tokens.*smaller max_length"):
        TextDataset(text, tokenizer, max_length=4)


@pytest.mark.parametrize("max_length", [0, -1, 2.5, True])
def test_invalid_max_length(tokenizer, max_length):
    with pytest.raises(ValueError, match="max_length must be a positive integer"):
        TextDataset("ABCDEFG", tokenizer, max_length=max_length)


@pytest.mark.parametrize("stride", [0, -1, 1.5, True, 5])
def test_invalid_stride(tokenizer, stride):
    with pytest.raises(ValueError, match="stride"):
        TextDataset("ABCDEFG", tokenizer, max_length=4, stride=stride)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size(tokenizer, batch_size):
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        create_dataloader("ABCDE", tokenizer, max_length=4, batch_size=batch_size)


def test_tokenizes_once_before_iteration(tokenizer):
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    loader = create_dataloader("ABCDEFGHIJ", tokenizer, max_length=4, batch_size=2)
    list(loader)
    list(loader)
    tokenizer.encode.assert_called_once_with("ABCDEFGHIJ")


def test_keeps_smaller_batch_separately_from_overlapping_tail(tokenizer):
    loader = create_dataloader(
        "ABCDEFGHIJ", tokenizer, max_length=4, batch_size=2, shuffle=False
    )
    batches = list(loader)
    assert [tuple(inputs.shape) for inputs, _ in batches] == [(2, 4), (1, 4)]
    assert [tuple(targets.shape) for _, targets in batches] == [(2, 4), (1, 4)]
    assert batches[0][0].tolist() == [list(b"ABCD"), list(b"EFGH")]
    assert batches[1][0].tolist() == [list(b"FGHI")]
    assert batches[1][1].tolist() == [list(b"GHIJ")]


def test_default_length_stride_and_batch_size(tokenizer):
    loader = create_dataloader("A" * 2305, tokenizer, shuffle=False)
    assert loader.dataset.max_length == loader.dataset.stride == 256
    assert len(loader.dataset) == 9
    assert len(loader) == 2
    assert [tuple(x.shape) for x, _ in loader] == [(8, 256), (1, 256)]


def test_shuffle_repeats_across_runs_but_advances_between_epochs(tokenizer):
    # Distinct first tokens identify examples without inspecting sampler internals.
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

    def make_loader():
        return create_dataloader(text, tokenizer, max_length=2, batch_size=4, seed=17)

    def epoch_order(loader):
        return [row[0].item() for inputs, _ in loader for row in inputs]

    rng_before = torch.random.get_rng_state().clone()
    first = make_loader()
    run_one = [epoch_order(first), epoch_order(first)]
    second = make_loader()
    run_two = [epoch_order(second), epoch_order(second)]
    assert run_one == run_two
    assert run_one[0] != run_one[1]
    assert sorted(run_one[0]) == sorted(run_one[1])
    assert len(set(run_one[0])) == len(first.dataset)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_saved_tokenizer_and_corpus_preserve_dialogue_and_unicode(tmp_path):
    corpus_path = tmp_path / "dialogue.json"
    corpus_path.write_text(
        '[{"Character": "LEIA", "Line": "Hello!"}, '
        '{"Character": "LUKE", "Line": "Hi, café🙂!"}]',
        encoding="utf-8",
    )
    model_path = Path(__file__).resolve().parents[1] / "models/star_wars_4096.bpe.json"
    tokenizer = BPETokenizer.load(model_path)
    text = load_corpus(corpus_path)
    expected = tokenizer.encode(text)
    original_merges = tokenizer.merges.copy()
    original_vocab = tokenizer.vocab.copy()

    loader = create_dataloader(
        text, tokenizer, max_length=len(expected) - 1, shuffle=False
    )
    inputs, targets = next(iter(loader))
    assert inputs[0].tolist() == expected[:-1]
    assert targets[0].tolist() == expected[1:]
    # Reassemble a complete source window before decoding: byte-level slices
    # can cut a Unicode character in half and need not decode independently.
    complete_window = inputs[0].tolist() + [targets[0, -1].item()]
    assert tokenizer.decode(complete_window) == "LEIA\nHello!\nLUKE\nHi, café🙂!"
    assert tokenizer.merges == original_merges
    assert tokenizer.vocab == original_vocab


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable")
def test_batch_can_be_transferred_by_training_code(tokenizer):
    loader = create_dataloader("ABCDE", tokenizer, max_length=4, shuffle=False)
    inputs, targets = next(iter(loader))
    assert inputs.device.type == targets.device.type == "cpu"
    assert torch.equal(inputs.to("cuda").cpu(), inputs)
    assert torch.equal(targets.to("cuda").cpu(), targets)
