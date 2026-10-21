"""Shared CPU batching for A6 QA training loaders."""

import torch
from torch.utils.data import DataLoader


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def seeded_loader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=0,
    collate_fn=None,
    *,
    num_workers=0,
    pin_memory=False,
):
    """Reproducible DataLoader used by answer_loader in trainer."""
    _require_positive_integer("batch_size", batch_size)
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
    )
