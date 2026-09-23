"""A byte-level Byte Pair Encoding tokenizer.

This replaces the word-level tokenizer in ``tokenizer.py``. That one splits text
on whitespace and punctuation and gives every distinct word its own ID, which
means any word it did not see while building the vocabulary turns into
``<|unk|>`` and is gone for good.

BPE fixes that by starting from the 256 possible byte values and repeatedly
merging whichever adjacent pair occurs most often in the training text, until
the vocabulary reaches the size we asked for. A word that was never seen during
training still encodes, because in the worst case it falls back to the
individual bytes it is made of.

Working on bytes rather than characters is what makes that a guarantee instead
of a hope: every possible input is a sequence of bytes, so there is no such
thing as an out-of-vocabulary character, and decoding is exact.

References used while writing this:
  - HuggingFace's tokenizer summary (the hug/pug/pun/bun worked example)
    https://huggingface.co/docs/transformers/tokenizer_summary
  - The GPT-2 byte pair encoder linked in the assignment
    https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/02_bonus_bytepair-encoder/bpe_openai_gpt2.py
"""

import json
import re
from collections import Counter
from itertools import pairwise

# Text is split on this before any merging happens, so that merges are never
# learned across a word boundary -- otherwise BPE would happily glue "the" and
# "Force" into a single token. This is the GPT-2 pattern from the reference
# implementation, rewritten to use Python's own `re` module: GPT-2 uses \p{L}
# and \p{N}, which need the third-party `regex` package, and [^\W\d_] / \d are
# the stdlib equivalents. Leading spaces are kept attached to the following
# word so that decoding can put them back exactly.
SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"""
)

END_OF_TEXT = "<|endoftext|>"
UNKNOWN = "<|unk|>"
SPECIAL_TOKENS = (END_OF_TEXT, UNKNOWN)

# The base vocabulary is one token per possible byte.
BYTE_VOCAB_SIZE = 256


def _count_pairs(word_freqs):
    """Count how often each adjacent pair of symbols occurs in the corpus.

    ``word_freqs`` maps a word (as a tuple of token IDs) to how many times that
    word appeared, so a pair inside a word that occurred 500 times counts 500
    times. Counting against the word frequencies instead of the raw text is
    what keeps training tractable: the Star Wars scripts are ~38k tokens but
    only a few thousand distinct words.
    """
    pairs = Counter()
    for word, freq in word_freqs.items():
        for pair in pairwise(word):
            pairs[pair] += freq
    return pairs


def _merge(symbols, pair, new_id):
    """Replace every occurrence of ``pair`` in ``symbols`` with ``new_id``."""
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Build one either by training on text::

        tokenizer = BPETokenizer()
        tokenizer.train(corpus, vocab_size=1000)

    or by loading merges that were trained earlier::

        tokenizer = BPETokenizer.load("vocab.json")
    """

    def __init__(self, merges=None, special_tokens=SPECIAL_TOKENS):
        # Maps a pair of existing token IDs to the new ID they merge into.
        # Insertion order is merge order, which is also priority order.
        self.merges = dict(merges or {})
        self.special_tokens = tuple(special_tokens)
        self._build_vocab()

    def _build_vocab(self):
        """Rebuild the ID -> bytes table from the current merge list."""
        # Every byte value is a token to start with.
        self.vocab = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        # Merges are applied in order, so each new token is always definable in
        # terms of tokens that already exist.
        for (first, second), new_id in self.merges.items():
            self.vocab[new_id] = self.vocab[first] + self.vocab[second]
        # Special tokens sit above the learned merges and are never produced by
        # merging, only by matching their literal text.
        next_id = BYTE_VOCAB_SIZE + len(self.merges)
        self.special_to_id = {}
        for offset, token in enumerate(self.special_tokens):
            token_id = next_id + offset
            self.special_to_id[token] = token_id
            self.vocab[token_id] = token.encode("utf-8")
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size=1000, verbose=False):
        """Learn merges from ``text`` until the vocabulary reaches ``vocab_size``.

        The vocabulary is 256 byte tokens, plus one token per merge learned,
        plus the special tokens -- so the number of merges to learn is just
        whatever is left over after accounting for the other two.

        ``vocab_size`` is an upper bound rather than a promise. A small corpus
        can run out of pairs to merge before the target is reached, in which
        case training stops there; padding the vocabulary out with junk tokens
        just to hit a number would be worse than returning a smaller one.
        """
        num_merges = vocab_size - BYTE_VOCAB_SIZE - len(self.special_tokens)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size must be at least "
                f"{BYTE_VOCAB_SIZE + len(self.special_tokens)} to fit the byte "
                f"tokens and special tokens, got {vocab_size}"
            )

        # Pre-tokenize, then represent each distinct word as a tuple of byte
        # values. Counting distinct words once is much cheaper than walking the
        # whole corpus on every merge.
        word_freqs = Counter()
        for chunk in SPLIT_PATTERN.findall(text):
            word_freqs[tuple(chunk.encode("utf-8"))] += 1
        word_freqs = dict(word_freqs)

        self.merges = {}
        for i in range(num_merges):
            pair_counts = _count_pairs(word_freqs)
            if not pair_counts:
                # The corpus has been merged down as far as it can go; every
                # word is a single token. Stop early rather than pad the vocab.
                break

            # Highest count wins. Ties are broken on the pair itself purely so
            # that training the same text twice gives the same merges.
            best_pair = max(pair_counts.items(), key=lambda item: (item[1], item[0]))[0]
            new_id = BYTE_VOCAB_SIZE + i
            self.merges[best_pair] = new_id
            word_freqs = {
                _merge(word, best_pair, new_id): freq
                for word, freq in word_freqs.items()
            }

            if verbose:
                print(f"merge {i + 1}/{num_merges}: {best_pair} -> {new_id}")

        self._build_vocab()
        return self

    def _encode_chunk(self, chunk):
        """Apply the learned merges to one pre-tokenized chunk of text."""
        ids = tuple(chunk.encode("utf-8"))
        while len(ids) >= 2:
            pairs = set(pairwise(ids))
            # Apply the earliest-learned merge that is still available. Merge
            # order matters: "es" has to exist before "est" can.
            candidates = [p for p in pairs if p in self.merges]
            if not candidates:
                break
            pair = min(candidates, key=lambda p: self.merges[p])
            ids = _merge(ids, pair, self.merges[pair])
        return list(ids)

    def encode(self, text):
        """Turn text into a list of token IDs."""
        if not text:
            return []

        # Split the special tokens out first and keep them whole. The old
        # tokenizer would shred a literal "<|endoftext|>" into punctuation.
        specials = "|".join(re.escape(token) for token in self.special_tokens)
        ids = []
        for part in re.split(f"({specials})", text):
            if not part:
                continue
            if part in self.special_to_id:
                ids.append(self.special_to_id[part])
            else:
                for chunk in SPLIT_PATTERN.findall(part):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        """Turn a list of token IDs back into text.

        Because every token is a byte string, decoding is just concatenation --
        no regex is needed to guess where the spaces went, which is why this
        round-trips exactly and the old tokenizer's decode did not.
        """
        parts = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise KeyError(f"unknown token id: {token_id}")
            parts.append(self.vocab[token_id])
        # errors="replace" only matters if a caller hands us a truncated
        # sequence that splits a multi-byte character; a full sequence produced
        # by encode() always decodes cleanly.
        return b"".join(parts).decode("utf-8", errors="replace")

    def save(self, path):
        """Write the merges to JSON so training does not have to be repeated."""
        payload = {
            "special_tokens": list(self.special_tokens),
            # JSON object keys must be strings, so pairs are stored as lists.
            "merges": [
                [first, second, new_id]
                for (first, second), new_id in self.merges.items()
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path):
        """Rebuild a tokenizer previously written by :meth:`save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        merges = {
            (first, second): new_id for first, second, new_id in payload["merges"]
        }
        return cls(merges=merges, special_tokens=tuple(payload["special_tokens"]))


def extract_strings(obj):
    """Collect every string in a nested JSON structure, depth first."""
    texts = []
    if isinstance(obj, str):
        texts.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_strings(item))
    elif isinstance(obj, dict):
        for value in obj.values():
            texts.extend(extract_strings(value))
    return texts


def load_corpus(path="SW_EpisodeIV_VI.json"):
    """Read the Star Wars script dataset into one string."""
    with open(path, "r", encoding="utf-8") as f:
        return " ".join(extract_strings(json.load(f)))


def measure(tokenizer, text):
    """Report the numbers used to argue this tokenizer beats the old one."""
    ids = tokenizer.encode(text)
    words = len(text.split())
    unk_id = tokenizer.special_to_id.get(UNKNOWN)
    return {
        "vocab_size": tokenizer.vocab_size,
        "tokens": len(ids),
        "tokens_per_word": len(ids) / words if words else 0.0,
        "unk_rate": sum(1 for i in ids if i == unk_id) / len(ids) if ids else 0.0,
        "round_trips": tokenizer.decode(ids) == text,
    }


def main():
    """Train on the Star Wars scripts and print how it compares."""
    corpus = load_corpus()
    # Hold out the last 10% so the unknown-token rate is measured on text the
    # merges were never trained on. On the training text it would be 0 by
    # definition, which proves nothing.
    split = int(len(corpus) * 0.9)
    train_text, held_out = corpus[:split], corpus[split:]

    tokenizer = BPETokenizer().train(train_text, vocab_size=1000)
    tokenizer.save("bpe_vocab.json")

    for name, text in (("training text", train_text), ("held-out text", held_out)):
        stats = measure(tokenizer, text)
        print(f"\n{name}:")
        print(f"  vocabulary size   {stats['vocab_size']}")
        print(f"  tokens            {stats['tokens']}")
        print(f"  tokens per word   {stats['tokens_per_word']:.2f}")
        print(f"  <|unk|> rate      {stats['unk_rate']:.2%}")
        print(f"  round trips       {stats['round_trips']}")


if __name__ == "__main__":  # pragma: no cover
    main()
