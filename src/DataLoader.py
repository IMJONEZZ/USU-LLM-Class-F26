import tiktoken
from torch.utils.data import DataLoader

from .TextDataset import TextDataset


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("gpt2")
    # Create dataset
    dataset = TextDataset(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
