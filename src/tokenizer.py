import json
import re

STARWARS_PATH = "./data/SW_EpisodeIV_VI.json"


def load_starwars_data():
    with open(STARWARS_PATH, "r") as f:
        starwars_data = json.load(f)
    preprocessed = [
        line
        for character_line in starwars_data
        for line in character_line["Line"].split()
    ]
    return preprocessed


def build_vocab(preprocessed):
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    return vocab


# Tokenize
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
