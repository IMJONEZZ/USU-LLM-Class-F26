import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import BPETokenizer, load_dialogue, train_bpe

DATA_PATH = "data/SW_EpisodeIV_VI.json"
SPECIAL_TOKEN = "<|endoftext|>"


def build_training_text(data_path=DATA_PATH, separator=SPECIAL_TOKEN):
    """
    Load the Star Wars dialogue dataset and join every line together into one
    long piece of text, inserting `separator` between each line so line
    boundaries are explicit rather than lines running directly into each other.
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    lines = [entry["Line"] for entry in data]
    return f" {separator} ".join(lines)


def encode_with_special_token(tokenizer, text, special_token=SPECIAL_TOKEN):
    """
    Encode text using the given BPE tokenizer, treating `special_token` as a
    single, indivisible token rather than letting it get broken down into
    subword pieces like ordinary text would be.

    We do this by splitting the text on the special token first, encoding
    each surrounding piece of real text normally, and manually inserting the
    special token's ID between pieces.
    """
    special_id = tokenizer.str_to_int[special_token]
    segments = text.split(special_token)

    ids = []
    for i, segment in enumerate(segments):
        segment = segment.strip()
        if segment:
            ids.extend(tokenizer.encode(segment))
        if i < len(segments) - 1:
            ids.append(special_id)
    return ids


class TextDataset(Dataset):
    """
    A PyTorch Dataset that turns a long stream of token IDs into fixed-length
    (input, target) chunks for next-token-prediction training, using a
    sliding window controlled by max_length and stride.
    """

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = encode_with_special_token(tokenizer, text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    text,
    tokenizer,
    batch_size=8,
    max_length=128,
    stride=32,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    """
    Build a TextDataset from `text` and wrap it in a PyTorch DataLoader.
    """
    dataset = TextDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


if __name__ == "__main__":  # pragma: no cover
    raw_text = load_dialogue(DATA_PATH)
    merges, vocab_symbols = train_bpe(raw_text, num_merges=300)
    tokenizer = BPETokenizer(merges, vocab_symbols)

    training_text = build_training_text()
    dataloader = create_dataloader(training_text, tokenizer)

    print(f"Number of batches: {len(dataloader)}")

    first_batch = next(iter(dataloader))
    inputs, targets = first_batch
    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)
    print("First input sequence (first 10 tokens):", inputs[0][:10].tolist())
    print("First target sequence (first 10 tokens):", targets[0][:10].tolist())
