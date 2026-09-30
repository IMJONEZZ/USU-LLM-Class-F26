"""Train and use byte-level BPE on all character names and dialogue in the corpus."""

import argparse
import json
from itertools import pairwise
from pathlib import Path

DEFAULT_VOCAB_SIZE = 4096


def text_to_byte_ids(text: str) -> list[int]:
    """Represent text as UTF-8 byte values, preserving its whitespace."""
    return list(text.encode("utf-8"))


def get_pair_counts(token_ids: list[int]) -> dict[tuple[int, int], int]:
    """Count adjacent pairs, including occurrences that overlap."""
    counts = {}
    for pair in pairwise(token_ids):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of a pair, scanning left to right.

    The caller supplies an unused ID for the merged token. The input list
    is left unchanged; this function only creates the new token sequence.
    """
    merged = []
    index = 0
    while index < len(token_ids):
        if (
            index + 1 < len(token_ids)
            and (token_ids[index], token_ids[index + 1]) == pair
        ):
            merged.append(new_id)
            index += 2  # Both tokens were consumed by the merge.
        else:
            merged.append(token_ids[index])
            index += 1
    return merged


def load_corpus(path: str | Path) -> str:
    """Read every Character and Line value, preserving text within each field."""
    with open(path, encoding="utf-8") as file:
        records = json.load(file)
    return "\n".join(
        value for record in records for value in (record["Character"], record["Line"])
    )


class BPETokenizer:
    """Learn ordered byte merges and preserve UTF-8 text without special tokens."""

    def __init__(self):
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = {}

    def train(self, text: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> list[int]:
        """Train from all input text and return its final token IDs.

        vocab_size includes the 256 starting byte tokens. Training may stop
        below this target if the entire sequence has fewer than two tokens.
        Calling train again starts a fresh vocabulary and merge history.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for the byte vocabulary")

        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        self.merges = {}
        token_ids = text_to_byte_ids(text)

        for _ in range(vocab_size - 256):
            counts = get_pair_counts(token_ids)
            if not counts:
                break

            # Prefer the largest count, then the smallest pair of token IDs.
            pair = min(counts, key=lambda pair: (-counts[pair], pair))
            new_id = len(self.vocab)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.merges[pair] = new_id  # Dictionaries retain insertion order.
            token_ids = merge_pair(token_ids, pair, new_id)

        return token_ids

    def encode(self, text: str) -> list[int]:
        """Apply learned merges in training order without changing the vocabulary.

        Before training, this returns the initial UTF-8 byte IDs. Unseen text
        remains representable because every byte has a token in the vocabulary.
        """
        token_ids = text_to_byte_ids(text)
        for pair, new_id in self.merges.items():
            if len(token_ids) < 2:
                break
            token_ids = merge_pair(token_ids, pair, new_id)
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Join token bytes before decoding, preserving the original text.

        Unknown IDs raise KeyError. Byte sequences that are not valid UTF-8
        raise UnicodeDecodeError instead of silently losing information.
        """
        data = b"".join(self.vocab[token_id] for token_id in token_ids)
        return data.decode("utf-8")

    def save(self, path: str | Path) -> None:
        """Save token bytes and ordered merge rules as a portable JSON file."""
        model = {
            "format_version": 1,
            # A token may contain only part of a UTF-8 character, so use hex.
            "vocab": [self.vocab[i].hex() for i in range(len(self.vocab))],
            "merges": [
                [left, right, new_id] for (left, right), new_id in self.merges.items()
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(model, file, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Restore a tokenizer without training or needing the original corpus.

        Validate the saved vocabulary against the ordered merge rules to
        catch files that would otherwise silently change token meanings.
        """
        with open(path, encoding="utf-8") as file:
            model = json.load(file)
        if not isinstance(model, dict) or model.get("format_version") != 1:
            raise ValueError("Unsupported tokenizer file format")
        if not isinstance(model.get("vocab"), list) or not isinstance(
            model.get("merges"), list
        ):
            raise TypeError("Tokenizer file must contain vocab and merges lists")
        try:
            saved_vocab = {
                i: bytes.fromhex(value) for i, value in enumerate(model["vocab"])
            }
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Vocabulary entries must be hexadecimal strings"
            ) from error

        tokenizer = cls()
        for rule in model["merges"]:
            if (
                not isinstance(rule, list)
                or len(rule) != 3
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in rule
                )
            ):
                raise ValueError("Each merge must contain three integer token IDs")
            left, right, new_id = rule
            pair = (left, right)
            if (
                new_id != len(tokenizer.vocab)
                or left not in tokenizer.vocab
                or right not in tokenizer.vocab
                or pair in tokenizer.merges
            ):
                raise ValueError("Merge rules contain invalid IDs or an invalid order")
            tokenizer.vocab[new_id] = tokenizer.vocab[left] + tokenizer.vocab[right]
            tokenizer.merges[pair] = new_id

        if tokenizer.vocab != saved_vocab:
            raise ValueError(
                "Saved vocabulary does not match the byte tokens and merge rules"
            )
        return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus",
        nargs="?",
        type=Path,
        help="JSON file containing Character/Line records (all records are used)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        help=f"target vocabulary size, including 256 byte tokens (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument("--save", type=Path, help="save the tokenizer to a JSON file")
    parser.add_argument(
        "--load", type=Path, help="load a saved tokenizer instead of training"
    )
    parser.add_argument("--text", help="text to encode and decode with the tokenizer")
    args = parser.parse_args()

    if args.load is not None:
        if args.corpus is not None or args.vocab_size is not None:
            parser.error("--load cannot be combined with a corpus or --vocab-size")
        tokenizer = BPETokenizer.load(args.load)
        print(f"Loaded tokenizer: {args.load}")
    else:
        corpus_path = (
            args.corpus
            or Path(__file__).resolve().parent.parent / "SW_EpisodeIV_VI.json"
        )
        vocab_size = DEFAULT_VOCAB_SIZE if args.vocab_size is None else args.vocab_size
        text = load_corpus(corpus_path)
        tokenizer = BPETokenizer()
        token_ids = tokenizer.train(text, vocab_size=vocab_size)
        print(f"Training characters: {len(text)}")
        print(f"Initial byte tokens: {len(text.encode('utf-8'))}")
        print(f"Tokens after training: {len(token_ids)}")
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Learned merges: {len(tokenizer.merges)}")
    if args.save is not None:
        tokenizer.save(args.save)
        print(f"Saved tokenizer: {args.save}")
    if args.text is not None:
        encoded = tokenizer.encode(args.text)
        print(f"Encoded IDs: {encoded}")
        print(f"Decoded text: {tokenizer.decode(encoded)!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
