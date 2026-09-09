import argparse
import os

from jsonExtract import process_json
from tokenizer import WordPieceTokenizer
from vocabBuild import build_wordpiece_vocab

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vocab-size",
    type=int,
    default=1000,
    help="Target vocabulary size (default: 1000)",
)
args = parser.parse_args()

# Load and process JSON
file_path = os.path.join(os.getcwd(), "SW_EpisodeIV_VI.json")

# Check that the training data exists before trying to process it.
if not os.path.isfile(file_path):
    print("Training data file was not found.")
    print(f"Expected file location: {file_path}")
else:
    all_text, preprocessed = process_json(file_path)

    # Build vocabulary
    vocab = build_wordpiece_vocab(
        preprocessed,
        target_vocab_size=args.vocab_size,
    )

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
