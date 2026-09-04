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
