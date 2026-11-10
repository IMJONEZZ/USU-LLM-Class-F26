import json

import torch
from torch.utils.data import DataLoader, Dataset

from src.bpe_tokenizer import BPETokenizer, train_bpe

DATA_PATH = "data/star_wars_script.jsonl"
SPECIAL_TOKEN = "<|endoftext|>"
VAL_MOVIE = "Revenge of the Sith"
TEST_MOVIE = "Return of the Jedi 4K83"


def load_scenes(path=DATA_PATH):
    """Load all scenes from the jsonl dataset file."""
    scenes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            scenes.append(json.loads(line))
    return scenes


def split_scenes_by_movie(scenes, val_movie=VAL_MOVIE, test_movie=TEST_MOVIE):
    """
    Split scenes into train/val/test based on the movie field, so that
    validation and test data come from entirely different films than
    training data -- preventing data leakage between the DataLoader and
    the Evaluator.
    """
    train, val, test = [], [], []
    for scene in scenes:
        movie = scene["movie"]
        if movie == val_movie:
            val.append(scene)
        elif movie == test_movie:
            test.append(scene)
        else:
            train.append(scene)
    return train, val, test


def scenes_to_text(scenes, separator=SPECIAL_TOKEN):
    """
    Flatten a list of scenes into one text blob. Lines within the same
    scene are joined with spaces; different scenes are joined with the
    special separator token, so the tokenizer/model can tell where one
    scene ends and the next begins.
    """
    scene_texts = []
    for scene in scenes:
        lines = [turn["text"] for turn in scene["turns"]]
        scene_texts.append(" ".join(lines))
    return f" {separator} ".join(scene_texts)


def encode_with_special_token(tokenizer, text, special_token=SPECIAL_TOKEN):
    """
    Encode text using the given BPE tokenizer, treating `special_token` as
    a single, indivisible token rather than letting it get broken down
    into subword pieces like ordinary text would be.
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
    """Build a TextDataset from `text` and wrap it in a PyTorch DataLoader."""
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
    scenes = load_scenes()
    train_scenes, val_scenes, test_scenes = split_scenes_by_movie(scenes)

    print(f"Train scenes: {len(train_scenes)}")
    print(f"Val scenes: {len(val_scenes)}")
    print(f"Test scenes: {len(test_scenes)}")

    train_text = scenes_to_text(train_scenes)
    val_text = scenes_to_text(val_scenes)
    test_text = scenes_to_text(test_scenes)

    merges, vocab_symbols = train_bpe(train_text, num_merges=300)
    tokenizer = BPETokenizer(merges, vocab_symbols)

    train_dataloader = create_dataloader(train_text, tokenizer)
    print(f"Number of training batches: {len(train_dataloader)}")
