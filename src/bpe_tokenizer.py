import itertools
from collections import Counter
from pathlib import Path

from src.tokenizer import (
    DATASET_PATH,
    END_OF_TEXT_TOKEN,
    UNKNOWN_TOKEN,
    load_dialogue,
)

Pair = tuple[int, int]
BASE_BYTE_COUNT = 256
END_OF_TEXT_ID = 256
UNKNOWN_ID = 257
MINIMUM_VOCAB_SIZE = 258


def count_pairs(token_ids: list[int]) -> Counter[Pair]:
    """Count adjacent token pairs."""
    return Counter(itertools.pairwise(token_ids))


def merge_pair(token_ids: list[int], pair: Pair, new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of a pair with a new ID."""
    merged = []
    index = 0

    while index < len(token_ids):
        pair_matches = (
            index + 1 < len(token_ids)
            and token_ids[index] == pair[0]
            and token_ids[index + 1] == pair[1]
        )

        if pair_matches:
            merged.append(new_id)
            index += 2
        else:
            merged.append(token_ids[index])
            index += 1

    return merged


class BytePairTokenizer:
    """A simplified byte-level Byte Pair Encoding tokenizer."""

    def __init__(self, target_vocab_size: int = 512) -> None:
        if target_vocab_size < MINIMUM_VOCAB_SIZE:
            raise ValueError(f"target_vocab_size must be at least {MINIMUM_VOCAB_SIZE}")

        self.target_vocab_size = target_vocab_size
        self.merges: dict[Pair, int] = {}
        self.id_to_bytes = self._initial_vocabulary()

    @staticmethod
    def _initial_vocabulary() -> dict[int, bytes]:
        vocabulary = {byte: bytes([byte]) for byte in range(BASE_BYTE_COUNT)}
        vocabulary[END_OF_TEXT_ID] = END_OF_TEXT_TOKEN.encode("utf-8")
        vocabulary[UNKNOWN_ID] = UNKNOWN_TOKEN.encode("utf-8")
        return vocabulary

    @property
    def vocabulary_size(self) -> int:
        """Return the current number of tokens."""
        return len(self.id_to_bytes)

    def train(self, text: str) -> None:
        """Learn frequent byte-pair merges from training text."""
        token_ids = list(text.encode("utf-8"))
        self.merges.clear()
        self.id_to_bytes = self._initial_vocabulary()
        next_id = MINIMUM_VOCAB_SIZE

        while next_id < self.target_vocab_size:
            pair_counts = count_pairs(token_ids)
            if not pair_counts:
                break

            best_pair = min(
                pair_counts,
                key=lambda pair: (-pair_counts[pair], pair[0], pair[1]),
            )
            if pair_counts[best_pair] < 2:
                break

            self.merges[best_pair] = next_id
            self.id_to_bytes[next_id] = (
                self.id_to_bytes[best_pair[0]] + self.id_to_bytes[best_pair[1]]
            )
            token_ids = merge_pair(token_ids, best_pair, next_id)
            next_id += 1

    def encode(self, text: str) -> list[int]:
        """Encode text using the learned merges."""
        token_ids = list(text.encode("utf-8"))

        for pair, merged_id in self.merges.items():
            token_ids = merge_pair(token_ids, pair, merged_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back into text."""
        try:
            encoded_text = b"".join(
                self.id_to_bytes[token_id] for token_id in token_ids
            )
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {error.args[0]}") from error

        return encoded_text.decode("utf-8")


def main(dataset_path: str | Path = DATASET_PATH) -> None:
    """Train BPE and demonstrate exact text reconstruction."""
    dialogue_lines = load_dialogue(dataset_path)
    training_text = "\n".join(dialogue_lines)

    tokenizer = BytePairTokenizer(target_vocab_size=512)
    tokenizer.train(training_text)

    sample = dialogue_lines[0]
    encoded = tokenizer.encode(sample)

    print(f"Vocabulary size: {tokenizer.vocabulary_size}")
    print(f"Learned merges: {len(tokenizer.merges)}")
    print(f"UTF-8 bytes: {len(sample.encode('utf-8'))}")
    print(f"BPE tokens: {len(encoded)}")
    print(f"Original: {sample}")
    print(f"Decoded: {tokenizer.decode(encoded)}")


if __name__ == "__main__":  # pragma: no cover
    main()
