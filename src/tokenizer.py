import json
import os
import re

# Load JSON
file_path = os.path.join(os.getcwd(), "SW_EpisodeIV_VI.json")

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# Recursively extract all strings from JSON
def extract_text(obj):
    texts = []

    if isinstance(obj, str):
        texts.append(obj)

    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_text(item))

    elif isinstance(obj, dict):
        for value in obj.values():
            texts.extend(extract_text(value))

    return texts


all_text = " ".join(extract_text(data))


# Preprocess / tokenize raw text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', all_text)

preprocessed = [item.strip() for item in preprocessed if item.strip()]


# Build vocab
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token: integer for integer, token in enumerate(all_tokens)}

print("Vocabulary size:", len(vocab))


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
        text = " ".join(self.int_to_str[i] for i in ids)

        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)

        return text


tokenizer = SimpleTokenizer(vocab)

ids = tokenizer.encode(all_text)

print("Token count:", len(ids))
print("First 100 token IDs:")
print(ids[:100])
