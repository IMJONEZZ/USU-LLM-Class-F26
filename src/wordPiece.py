import os

from jsonExtract import process_json
from tokenizer import WordPieceTokenizer
from vocabBuild import build_wordpiece_vocab

# Load and process JSON
file_path = os.path.join(os.getcwd(), "SW_EpisodeIV_VI.json")

all_text, preprocessed = process_json(file_path)


# Build vocabulary
vocab = build_wordpiece_vocab(preprocessed, target_vocab_size=1000)

print("Vocabulary size:", len(vocab))


# Create tokenizer
tokenizer = WordPieceTokenizer(vocab)

# Encode text
ids = tokenizer.encode(all_text)

print("Token count:", len(ids))

print("First 100 token IDs:")
print(ids[:100])


# Decode sample
decoded = tokenizer.decode(ids[:100])

print("\nDecoded sample:")
print(decoded)
