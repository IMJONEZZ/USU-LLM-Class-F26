import json
import re
from pathlib import Path

data_path = Path(__file__).resolve().parent.parent / "SW_EpisodeIV_VI.json"

# preprocess by splitting all text by whitespace and keeping only unique values
with data_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

preprocessed = []

for snippet in data:
    split = re.split(r'([,.:;?_!"()\']|--|\s)', snippet["Line"])
    split = [item.strip() for item in split if item.strip()]
    for item in split:
        if item not in preprocessed:
            preprocessed.append(item)

# Build Vocab
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))


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
