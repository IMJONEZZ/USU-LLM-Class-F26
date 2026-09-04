# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///


__generated_with = "0.24.0"

# %%
import re
import string
from collections import Counter
from itertools import pairwise

from datasets import load_dataset

# uvx marimo export script marimo_notebooks/tokenizer.py -o src/tokenizer.py

# %%
ds = load_dataset("andrewkroening/Star-wars-scripts-dialogue-IV-VI")
print(ds)

# %%
# Get all rows from the training split
rows = ds["train"][:]["Line"]
rows[:5]  # Display the first 5 rows

# %%
full_document_text = " <|endoftext|> ".join(rows)
full_document_text[:1000]  # Display the first 1000 characters of the full document text

# %%
preprocessed = re.split(rf"({re.escape(string.punctuation)}|--|\s)", full_document_text)
preprocessed = [
    item.strip() for item in preprocessed if item.strip()
]  # Remove empty strings and whitespace
preprocessed[:10]  # Display the first 100 tokens of the preprocessed text

# %%
# Build Vocab
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}
print(len(vocab.items()))


# %%
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


# %%
tokenizer = SimpleTokenizer(vocab)
encoded = tokenizer.encode(
    "I love using my light saber to kill Darth Vader"
)  # Example usage of the tokenizer
decoded = tokenizer.decode(encoded)  # Decode the encoded tokens back to text
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")


# %%
class BPETokenizer:
    """A simple BPE tokenizer."""

    def __init__(self, corpus, vocab_size=1000):
        # Placeholder for BPE logic
        self.corpus = corpus
        self.vocab_size = vocab_size
        self.special_token = "<|endoftext|>"
        self.unknown_token = "<|unk|>"
        self._create_vocab()  # creates self.str_to_int, self.int_to_str, and self.merge_rules

    def _split_into_initial_tokens(self, text):
        """Split text into characters while preserving special tokens."""
        parts = re.split(
            f"({re.escape(self.special_token)})",
            text,
        )
        tokens = []
        for part in parts:
            if not part:
                continue
            if part == self.special_token:
                tokens.append(part)
            else:
                tokens.extend(part)
        return tokens

    def _merge_pair(self, tokens, pair):
        """Merge all non-overlapping occurrences of pair."""
        merged_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i + 1 < len(tokens)
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                merged_tokens.append("".join(pair))
                i += 2
            else:
                merged_tokens.append(tokens[i])
                i += 1
        return merged_tokens

    def _create_vocab(self):
        # Placeholder for creating BPE vocab
        tokens = self._split_into_initial_tokens(self.corpus)
        vocab = sorted(set(tokens) | {self.special_token, self.unknown_token})
        self.merge_rules = []
        while len(vocab) < self.vocab_size:
            counts = Counter(
                pair
                for pair in pairwise(tokens)
                if self.special_token
                not in pair  # Don't merge the special token with anything else.
            )

            if not counts:  # we've run out of pairs to merge
                break

            most_common_pair = counts.most_common(1)[0][0]
            new_token = "".join(most_common_pair)
            self.merge_rules.append(most_common_pair)
            if new_token not in vocab:
                vocab.append(new_token)

            tokens = self._merge_pair(tokens, most_common_pair)

        self.str_to_int = {token: i for i, token in enumerate(vocab)}
        self.int_to_str = {i: token for token, i in self.str_to_int.items()}

    def encode(self, text):
        tokens = self._split_into_initial_tokens(text)
        tokens = [
            token if token in self.str_to_int else self.unknown_token
            for token in tokens
        ]

        for pair in self.merge_rules:
            tokens = self._merge_pair(tokens, pair)

        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids):
        return "".join(self.int_to_str[token_id] for token_id in ids)


# %%
bpe = BPETokenizer("<|endoftext|>".join(rows))
print(list(reversed(bpe.str_to_int.items()))[:10])

# %%
encoded_bpe = bpe.encode(
    "I love using my light saber to kill Darth Vader"
)  # Example usage of the tokenizer
decoded_bpe = bpe.decode(encoded_bpe)  # Decode the encoded tokens back to text
print(f"Encoded: {encoded_bpe}")
print(f"Decoded: {decoded_bpe}")

# %%
encoded_bpe2 = bpe.encode(
    "Supercalifragilisticexpialidocious"
)  # Example usage of the tokenizer
decoded_bpe2 = bpe.decode(encoded_bpe2)  # Decode the encoded tokens back to text
print(f"Encoded: {encoded_bpe2}")
print(f"Decoded: {decoded_bpe2}")
