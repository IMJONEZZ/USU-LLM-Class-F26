"""Prepare next-token examples and reproducible batches from tokenized text."""

import torch
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import BPETokenizer


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class TextDataset(Dataset):
    """Encode text once and retrieve fixed-length input/target pairs on the CPU.

    Regular input windows do not overlap when stride is omitted. A final
    window may overlap to retain the tail without padding. Returned tensors
    are views of the stored tokens and should not be modified in place.
    """

    def __init__(
        self,
        text: str,
        tokenizer: BPETokenizer,
        max_length: int = 256,
        stride: int | None = None,
    ):
        _require_positive_integer("max_length", max_length)
        if stride is None:
            stride = max_length
        _require_positive_integer("stride", stride)
        if stride > max_length:
            raise ValueError("stride must not exceed max_length; this would skip data")

        token_ids = tokenizer.encode(text)
        if len(token_ids) < max_length + 1:
            raise ValueError(
                f"Need at least {max_length + 1} tokens for max_length={max_length}; "
                f"got {len(token_ids)}. Use more text or a smaller max_length."
            )

        self.max_length = max_length
        self.stride = stride
        self.token_ids = torch.tensor(token_ids, dtype=torch.long, device="cpu")

        # Each pair needs one extra source token for the final target.
        last_start = len(token_ids) - max_length - 1
        self.starts = list(range(0, last_start + 1, stride))
        if self.starts[-1] != last_start:
            self.starts.append(last_start)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.max_length
        return self.token_ids[start:end], self.token_ids[start + 1 : end + 1]


def create_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    max_length: int = 256,
    stride: int | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Build a loader that retains the final smaller batch.

    A private generator makes shuffle order reproducible across newly created
    loaders with the same seed. Its state advances between epochs, producing
    fresh orders without resetting the seed or changing the global RNG state.
    Device selection and GPU transfers belong to the future training loop.
    """
    _require_positive_integer("batch_size", batch_size)
    dataset = TextDataset(text, tokenizer, max_length=max_length, stride=stride)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )
