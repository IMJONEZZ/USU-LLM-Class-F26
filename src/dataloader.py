import truststore

truststore.inject_into_ssl()

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


def _group_lines_by_char_count(lines, target_chars):
    blocks = []
    current_lines = []
    current_length = 0

    for line in lines:
        current_lines.append(line)
        current_length += len(line)

        if current_length >= target_chars:
            blocks.append("\n".join(current_lines))
            current_lines = []
            current_length = 0

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


def _chunk_token_ids(token_ids, max_length, stride):
    input_chunks = []
    target_chunks = []

    for i in range(0, len(token_ids) - max_length, stride):
        input_chunks.append(token_ids[i : i + max_length])
        target_chunks.append(token_ids[i + 1 : i + max_length + 1])

    return input_chunks, target_chunks


class _Gpt2CollateFn:
    def __init__(self, max_length, stride):
        self.max_length = max_length
        self.stride = stride
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __call__(self, batch):
        text = "\n".join(batch)
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        input_chunks, target_chunks = _chunk_token_ids(
            token_ids, self.max_length, self.stride
        )

        input_batch = torch.stack([torch.tensor(chunk) for chunk in input_chunks])
        target_batch = torch.stack([torch.tensor(chunk) for chunk in target_chunks])
        return input_batch, target_batch


class TextDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        input_chunks, target_chunks = _chunk_token_ids(token_ids, max_length, stride)

        self.input_ids = [torch.tensor(chunk) for chunk in input_chunks]
        self.target_ids = [torch.tensor(chunk) for chunk in target_chunks]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = TextDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return dataloader


class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]


def _make_gpt2_collate_fn(tokenizer, max_length, stride):
    def collate_fn(batch):
        text = "\n".join(batch)
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        input_chunks, target_chunks = _chunk_token_ids(token_ids, max_length, stride)

        input_batch = torch.stack([torch.tensor(chunk) for chunk in input_chunks])
        target_batch = torch.stack([torch.tensor(chunk) for chunk in target_chunks])
        return input_batch, target_batch

    return collate_fn


def create_dataloader_v2(
    lines,
    max_length=256,
    stride=128,
    target_chars=None,
):
    if target_chars is None:
        target_chars = 400

    blocks = _group_lines_by_char_count(lines, target_chars)
    dataset = LineDataset(blocks)
    collate_fn = _Gpt2CollateFn(max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        in_order=False,
        drop_last=True,
    )
    return dataloader
