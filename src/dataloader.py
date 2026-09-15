import torch
from torch.utils.data import DataLoader, Dataset

from src.tokenizer import SentencePieceTokenizer


class StarWarsDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        # Use a sliding window to chunk the tokenized text into overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            input_seq = token_ids[i : i + max_length]
            target_seq = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_seq))
            self.target_ids.append(torch.tensor(target_seq))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    batch_size=16,
    max_length=128,
    stride=64,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    # Initialize the tokenizer
    tokenizer = SentencePieceTokenizer()

    # Text should be a single string. If a list of strings, join them into one string.
    joined_text = " ".join(text) if isinstance(text, list) else text

    # Custom tokenizer requires number of merges to be specified.
    tokenizer.train(joined_text, num_merges=300)

    # Create the dataset
    dataset = StarWarsDataset(joined_text, tokenizer, max_length, stride)

    # Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader
