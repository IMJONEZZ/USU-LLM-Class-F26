"""Baseline word-level tokenizer from the previous homework assignment.

This is the lossless whitespace tokenizer (SimpleTokenizerV3): whitespace is
kept as an explicit token, punctuation is split into its own pre-tokens, and
decode() simply concatenates, so encode/decode round trips exactly. Unknown
words map to the <|unk|> token.
"""

import re

# Keep commas, periods, colons, etc. as their own tokens, keep "--", and keep
# every whitespace character as a token so decoding can restore the original.
SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'

UNK = "<|unk|>"


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index: token for token, index in vocab.items()}

    @classmethod
    def train(cls, training_text):
        """Build the vocab from a corpus: every distinct piece + <|unk|>."""
        tokens = {token for token in re.split(SPLIT_PATTERN, training_text) if token}
        vocab = {token: index for index, token in enumerate(sorted(tokens))}
        vocab[UNK] = len(vocab)
        return cls(vocab)

    def encode(self, text):
        pieces = [piece for piece in re.split(SPLIT_PATTERN, text) if piece != ""]
        return [self.str_to_int.get(piece, self.str_to_int[UNK]) for piece in pieces]

    def decode(self, ids):
        return "".join(self.int_to_str[index] for index in ids)
