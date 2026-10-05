import json
import re
from pathlib import Path

END_OF_TEXT_TOKEN = "<|endoftext|>"
UNKNOWN_TOKEN = "<|unk|>"
DATASET_PATH = Path("data/SW_EpisodeIV_VI.json")


def split_text(text: str) -> list[str]:
    """Split text into words and punctuation tokens."""
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [item.strip() for item in preprocessed if item.strip()]


def load_dialogue(path: str | Path) -> list[str]:
    """Load only dialogue lines from the Star Wars dataset."""
    with Path(path).open(encoding="utf-8") as file:
        records = json.load(file)

    return [record["Line"] for record in records]


def build_vocab(text: str) -> dict[str, int]:
    """Build a case-preserving vocabulary from text."""
    all_tokens = sorted(set(split_text(text)))
    all_tokens.extend([END_OF_TEXT_TOKEN, UNKNOWN_TOKEN])
    return {token: integer for integer, token in enumerate(all_tokens)}


class SimpleTokenizer:
    """Convert text tokens to integer IDs and back."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self.str_to_int = vocab
        self.int_to_str = {integer: token for token, integer in vocab.items()}

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        preprocessed = split_text(text)
        preprocessed = [
            item if item in self.str_to_int else UNKNOWN_TOKEN for item in preprocessed
        ]
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back into readable text."""
        text = " ".join(self.int_to_str[token_id] for token_id in ids)
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)


def main() -> None:
    """Build the tokenizer and demonstrate it on the first dialogue line."""
    dialogue_lines = load_dialogue(DATASET_PATH)
    training_text = " ".join(dialogue_lines)
    vocab = build_vocab(training_text)
    tokenizer = SimpleTokenizer(vocab)

    sample = dialogue_lines[0]
    encoded = tokenizer.encode(sample)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Original: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")


if __name__ == "__main__":  # pragma: no cover
    main()
