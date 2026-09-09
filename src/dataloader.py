__generated_with = "0.24.0"

# %%
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

# uvx --link-mode=copy marimo export script marimo_notebooks/dataloader.py -o src/dataloader.py

# %%
ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
rows = ds["train"][:]["Line"]
full_document_text = " <|endoftext|> ".join(rows)

print(full_document_text[:1000])

# %%
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})


# %%
class TextDataset(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer,
        max_length: int = 512,
        step: int = 256,
        pad: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.step = step
        self.pad = pad

        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        self.chunks = self._create_chunks(token_ids)

    def _create_chunks(self, token_ids: list[int]) -> list[list[int]]:

        chunks = []
        window_size = self.max_length + 1

        for i in range(0, len(token_ids), self.step):
            chunk = token_ids[i : i + window_size]
            reached_end = i + window_size >= len(token_ids)

            if len(chunk) < window_size:
                if self.pad:
                    chunk += [self.pad_token_id] * (window_size - len(chunk))
                else:
                    break

            chunks.append(chunk)

            if reached_end:
                break

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.chunks[index]
        input_chunk = chunk[:-1]
        target_chunk = chunk[-1]
        return torch.tensor(input_chunk), torch.tensor(target_chunk)


# %%
dataset = TextDataset(
    text=full_document_text,
    tokenizer=tokenizer,
    max_length=256,
    step=128,
    pad=True,
)
print(f"Dataset size: {len(dataset)}")
sample = dataset[0]
print(f"Sample input: {sample[0][:5]}")
print(f"Sample target: {sample[1]}")


# %%
def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


# %%
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)
batch = next(iter(dataloader))
print(f"Batch input shape: {batch[0].shape}")
print(f"Batch target shape: {batch[1].shape}")
