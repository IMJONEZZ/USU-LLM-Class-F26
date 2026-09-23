import torch
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import END_OF_TEXT_ID, BytePairTokenizer


class DialogueDataset(Dataset):
    """Prepare fixed-length token sequences for next-token prediction."""

    def __init__(
        self,
        dialogue_lines: list[str],
        tokenizer: BytePairTokenizer,
        sequence_length: int = 32,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.sequence_length = sequence_length

        token_ids: list[int] = []
        for line in dialogue_lines:
            token_ids.extend(tokenizer.encode(line))
            token_ids.append(END_OF_TEXT_ID)

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        """Count complete samples, reserving one token for the final target."""
        return max(0, (len(self.tokens) - 1) // self.sequence_length)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an input sequence and its one-token-shifted target."""
        if index < 0 or index >= len(self):
            raise IndexError("Sample index out of range")

        start = index * self.sequence_length
        end = start + self.sequence_length

        inputs = self.tokens[start:end]
        targets = self.tokens[start + 1 : end + 1]

        return inputs, targets


def create_dataloader(
    dialogue_lines: list[str],
    tokenizer: BytePairTokenizer,
    sequence_length: int = 32,
    batch_size: int = 4,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Batch dialogue samples, with optional reproducible shuffling."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dataset = DialogueDataset(dialogue_lines, tokenizer, sequence_length)

    if len(dataset) == 0:
        raise ValueError("Not enough tokens to create a complete sample")

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=False,
        num_workers=0,
    )
