import pytest
import torch

from src.bpe_tokenizer import END_OF_TEXT_ID, BytePairTokenizer
from src.dataloader import DialogueDataset, create_dataloader


@pytest.fixture
def tokenizer():
    """Use byte tokens to make expected results easy to check."""
    return BytePairTokenizer()


def test_equal_sequence_lengths(tokenizer):
    dataset = DialogueDataset(["abcdefghij"], tokenizer, sequence_length=3)

    assert len(dataset) == 3
    for inputs, targets in dataset:
        assert inputs.shape == (3,)
        assert targets.shape == (3,)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long


def test_targets_are_next_tokens(tokenizer):
    dataset = DialogueDataset(["abcdef"], tokenizer, sequence_length=3)

    inputs, targets = dataset[0]
    assert inputs.tolist() == [97, 98, 99]
    assert targets.tolist() == [98, 99, 100]

    inputs, targets = dataset[1]
    assert inputs.tolist() == [100, 101, 102]
    assert targets.tolist() == [101, 102, END_OF_TEXT_ID]


def test_dialogue_boundaries(tokenizer):
    dataset = DialogueDataset(["ab", "cd"], tokenizer, sequence_length=2)

    assert dataset.tokens.tolist() == [97, 98, END_OF_TEXT_ID, 99, 100, END_OF_TEXT_ID]


def test_incomplete_final_chunk_is_skipped(tokenizer):
    dataset = DialogueDataset(["abcdefgh"], tokenizer, sequence_length=3)

    assert len(dataset) == 2
    inputs, targets = dataset[1]
    assert inputs.tolist() == [100, 101, 102]
    assert targets.tolist() == [101, 102, 103]

    with pytest.raises(IndexError):
        dataset[2]


@pytest.mark.parametrize("lines", [[], ["a"]])
def test_insufficient_data(tokenizer, lines):
    dataset = DialogueDataset(lines, tokenizer, sequence_length=3)
    assert len(dataset) == 0

    with pytest.raises(ValueError, match="Not enough tokens"):
        create_dataloader(lines, tokenizer, sequence_length=3)


@pytest.mark.parametrize("sequence_length", [0, -1])
def test_invalid_sequence_length(tokenizer, sequence_length):
    with pytest.raises(ValueError, match="sequence_length"):
        DialogueDataset(["abcdef"], tokenizer, sequence_length)


@pytest.mark.parametrize("index", [-1, 2])
def test_invalid_sample_index(tokenizer, index):
    dataset = DialogueDataset(["abcdef"], tokenizer, sequence_length=3)

    with pytest.raises(IndexError):
        dataset[index]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(tokenizer, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        create_dataloader(["abcdef"], tokenizer, batch_size=batch_size)


def test_batching_keeps_final_smaller_batch(tokenizer):
    loader = create_dataloader(
        ["abcdefghij"],
        tokenizer,
        sequence_length=3,
        batch_size=2,
        shuffle=False,
    )
    batches = list(loader)

    assert len(batches) == 2
    assert batches[0][0].shape == (2, 3)
    assert batches[0][1].shape == (2, 3)
    assert batches[1][0].shape == (1, 3)
    assert batches[1][1].shape == (1, 3)
    assert batches[0][0][0].tolist() == [97, 98, 99]
    assert batches[1][0][0].tolist() == [103, 104, 105]


def collect_samples(loader):
    """Keep inputs and targets together when comparing sample order."""
    return [
        (tuple(inputs.tolist()), tuple(targets.tolist()))
        for input_batch, target_batch in loader
        for inputs, targets in zip(input_batch, target_batch, strict=True)
    ]


def test_shuffle_preserves_samples_and_targets(tokenizer):
    lines = ["abcdefghijklmnopqrstuvwxyz"]
    ordered = create_dataloader(
        lines, tokenizer, sequence_length=2, batch_size=4, shuffle=False
    )
    shuffled = create_dataloader(
        lines, tokenizer, sequence_length=2, batch_size=4, seed=42
    )

    expected = collect_samples(ordered)
    actual = collect_samples(shuffled)

    assert sorted(actual) == sorted(expected)


def test_seed_reproduces_multiple_epochs(tokenizer):
    lines = ["abcdefghijklmnopqrstuvwxyz"]
    first_loader = create_dataloader(
        lines, tokenizer, sequence_length=2, batch_size=4, seed=42
    )
    second_loader = create_dataloader(
        lines, tokenizer, sequence_length=2, batch_size=4, seed=42
    )

    first_epoch = collect_samples(first_loader)
    second_epoch = collect_samples(first_loader)

    assert first_epoch == collect_samples(second_loader)
    assert second_epoch == collect_samples(second_loader)
    assert first_epoch != second_epoch


def test_trained_bpe_integration():
    lines = ["abababab", "abababab"]
    tokenizer = BytePairTokenizer(target_vocab_size=260)
    tokenizer.train("\n".join(lines))

    expected = []
    for line in lines:
        expected.extend(tokenizer.encode(line))
        expected.append(END_OF_TEXT_ID)

    assert tokenizer.merges

    dataset = DialogueDataset(lines, tokenizer, sequence_length=2)
    assert dataset.tokens.tolist() == expected

    for index in range(len(dataset)):
        start = index * 2
        inputs, targets = dataset[index]
        assert inputs.tolist() == expected[start : start + 2]
        assert targets.tolist() == expected[start + 1 : start + 3]
