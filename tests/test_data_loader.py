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
