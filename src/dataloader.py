"""PyTorch dataset utilities for next-token language-model training."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

try:
    from src.BPEtokenizer import BPE_Algorithm, BPE_Encoder, BPE_Vocabulary
except ModuleNotFoundError:
    from BPEtokenizer import BPE_Algorithm, BPE_Encoder, BPE_Vocabulary


class TextDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        txt: str,
        tokenizer: BPE_Encoder,
        max_length: int = 512,
        stride: int = 128,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Build a batched next-token dataloader using a BPE tokenizer."""
    algorithm = BPE_Algorithm(training_text=txt, merge_size=5000)
    algorithm.mergify()
    vocabulary = BPE_Vocabulary(algorithm.bpe_merges)
    tokenizer = BPE_Encoder(vocabulary.str2id)

    dataset = TextDataset(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "SW_EpisodeIV_VI.json"
    dialogue = json.loads(data_path.read_text(encoding="utf-8"))

    text = BPE_Vocabulary.end_of_text_token.join(entry["Line"] for entry in dialogue)
    dataloader = create_dataloader_v1(text, batch_size=64, max_length=512, stride=128)
    print(
        f"Created {len(dataloader.dataset)} training examples in {len(dataloader)} batches."
    )
