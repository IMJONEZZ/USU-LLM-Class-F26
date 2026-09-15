import json
import re

DATA_PATH = "data/SW_EpisodeIV_VI.json"


def load_dialogue(path):
    """Load the dataset and join all dialogue lines into one text blob."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lines = [entry["Line"] for entry in data]
    return "\n".join(lines)


def build_vocab(text):
    """Build a vocabulary dict from raw text."""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]

    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


if __name__ == "__main__":  # pragma: no cover
    raw_text = load_dialogue(DATA_PATH)
    vocab = build_vocab(raw_text)
    print(len(vocab.items()))

    tokenizer = SimpleTokenizer(vocab)
    sample_text = "Did you hear that? We're doomed!"
    ids = tokenizer.encode(sample_text)
    print(ids)
    print(tokenizer.decode(ids))
