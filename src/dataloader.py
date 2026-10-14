"""Shared CPU batching for A6 QA training loaders."""

import torch
from torch.utils.data import DataLoader


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def seeded_loader(dataset, batch_size=1, shuffle=False, seed=0, collate_fn=None):
    """Reproducible DataLoader used by answer_loader in trainer."""
    _require_positive_integer("batch_size", batch_size)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )
