import torch

from src.DataLoader import create_dataloader_v1


class FakeTokenizer:
    def encode(self, txt, allowed_special=None):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8]


def test_dataloader_returns_expected_inputs_and_targets():
    dataloader = create_dataloader_v1(
        txt="unused because tokenizer is fake",
        tokenizer=FakeTokenizer(),
        batch_size=2,
        max_length=4,
        stride=4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    batches = list(dataloader)

    # Windows:
    # input  [0, 1, 2, 3] -> target [1, 2, 3, 4]
    # input  [4, 5, 6, 7] -> target [5, 6, 7, 8]
    assert len(batches) == 1

    inputs, targets = batches[0]

    expected_inputs = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    expected_targets = torch.tensor(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    assert torch.equal(inputs, expected_inputs)
    assert torch.equal(targets, expected_targets)


def test_dataloader_includes_last_valid_window():
    class EndBoundaryTokenizer:
        def encode(self, txt, allowed_special=None):
            # Exactly enough for two input/target windows.
            return [0, 1, 2, 3, 4, 5, 6, 7, 8]

    dataloader = create_dataloader_v1(
        txt="unused",
        tokenizer=EndBoundaryTokenizer(),
        batch_size=2,
        max_length=4,
        stride=4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    inputs, targets = next(iter(dataloader))

    # Specifically verify the final valid window.
    assert torch.equal(inputs[-1], torch.tensor([4, 5, 6, 7]))
    assert torch.equal(targets[-1], torch.tensor([5, 6, 7, 8]))


def test_dataloader_does_not_create_incomplete_window():
    class ShortTokenizer:
        def encode(self, txt, allowed_special=None):
            # Enough for one complete window, but not two.
            return [0, 1, 2, 3, 4, 5, 6, 7]

    dataloader = create_dataloader_v1(
        txt="unused",
        tokenizer=ShortTokenizer(),
        batch_size=2,
        max_length=4,
        stride=4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    batches = list(dataloader)

    assert len(batches) == 1

    inputs, targets = batches[0]

    assert inputs.shape == (1, 4)
    assert targets.shape == (1, 4)

    assert torch.equal(inputs[0], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(targets[0], torch.tensor([1, 2, 3, 4]))


def test_dataloader_returns_empty_when_not_enough_tokens():
    class TooShortTokenizer:
        def encode(self, txt, allowed_special=None):

            return [0, 1, 2, 3]

    dataloader = create_dataloader_v1(
        txt="unused",
        tokenizer=TooShortTokenizer(),
        batch_size=2,
        max_length=4,
        stride=4,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    assert list(dataloader) == []
