"""Qwen3-style tokenizer: byte-level BPE with the real Qwen3 pre-tokenizer.

This is the "better tokenizer" for this assignment: it ports the design of
the tokenizer Qwen3 actually ships (verified from Qwen/Qwen3-32B's
tokenizer_config.json + config.json: vocab_size 151,936, Qwen2Tokenizer,
byte-level BPE) down to a small, trainable, stdlib-only implementation.

Design points it gains over the baseline SimpleTokenizer:

1. Byte-level, byte-faithful: text is UTF-8 encoded, each byte is remapped to
   a printable unicode char (GPT-2's bytes_to_unicode, space -> "Ġ"), and BPE
   merges operate on those symbols. Any string of any language round trips
   bit-exactly — there is no <|unk|> and there can be none.
2. Qwen3's pre-tokenizer regex: letter runs may carry one non-letter prefix
   (",world" and "_foo" form), digits are split ONE per pre-token (arithmetic
   aid), and other punctuation can absorb trailing newlines. Nothing merges
   across pre-token boundaries.
3. Special tokens as first-class vocabulary entries (Qwen3's real set:
    and the <|im_*> turn delimiters, the FIM family for code infilling, and
   <|repo_name|>/<|file_sep|> from Qwen2.5-Coder). Encoding them out of user
   text requires an allow-list, tiktoken-style.

Train-on-corpus API (mirrors SimpleTokenizer.train):

    tok = Qwen3Tokenizer.train(corpus, vocab_size=300)
    ids = tok.encode("Hello, world")
    tok.decode(ids) == "Hello, world"
"""

import re
import unicodedata
from collections import Counter
from functools import cache
from itertools import pairwise

# The end-of-text token. Assembled from pieces so no copy/paste or editor
# transport can silently strip it into plain, indistinguishable whitespace.
EOT = "<|" + "endoftext" + "|>"

# ---------------------------------------------------------------------------
# BPE engine — symbols are plain strings, so the same trainer serves anything.
# ---------------------------------------------------------------------------


def _pair_counts(sequences):
    counts = Counter()
    for seq in sequences:
        for pair in pairwise(seq):
            counts[pair] += 1
    return counts


def _merge_pair(seq, pair, merged):
    out, i = [], 0
    while i < len(seq):
        if seq[i] == pair[0] and i + 1 < len(seq) and seq[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def bpe_train(sequences, num_merges):
    sequences = [list(s) for s in sequences]
    merges = []
    for _ in range(num_merges):
        counts = _pair_counts(sequences)
        if not counts:
            break
        top = max(counts, key=counts.get)
        if counts[top] == 1:
            break
        merged = top[0] + top[1]
        sequences = [_merge_pair(seq, top, merged) for seq in sequences]
        merges.append((top, merged))
    return merges


def bpe_encode(sequence, merges):
    symbols = list(sequence)
    for (a, b), merged in merges:
        symbols = _merge_pair(symbols, (a, b), merged)
    return symbols


def _compose_vocab(base_symbols, merges):
    vocab = {s: i for i, s in enumerate(base_symbols)}
    for _, merged in merges:
        vocab[merged] = len(vocab)
    return vocab


# ---------------------------------------------------------------------------
# Byte machinery — GPT-2/Qwen map every byte to a printable unicode char so
# BPE has full UTF-8 coverage and decode is a pure byte write-back.
# ---------------------------------------------------------------------------


@cache
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@cache
def _unicode_classes():
    """One sweep over the Unicode tables for the char classes Qwen3 needs.

    The stdlib `re` has no \\p{L}/\\p{N}, so the pattern can't reference
    letter/number classes directly — and interpolating one bracket class
    inside another would close the outer class early. Instead, all four
    classes are built as standalone, pre-escaped (\\U...) bracket classes:

        L      letters (\\p{L})
        N      numbers (\\p{N})
        PRE    chars allowed to prefix a letter run: everything but \\r, \\n,
               letters and numbers (Qwen3's [^\\r\\n\\p{L}\\p{N}])
        PUNCT  chars in a punctuation run: everything but whitespace,
               letters and numbers (Qwen3's [^\\s\\p{L}\\p{N}])

    Runs once per process (~1.1M code points).
    """
    letters, numbers, prefix, punct = [], [], [], []
    cr_lf = (ord("\r"), ord("\n"))
    for cp in range(0x110000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        letter = cat.startswith("L")
        number = cat.startswith("N")
        if letter:
            letters.append(cp)
        if number:
            numbers.append(cp)
        if not (letter or number or cp in cr_lf):
            prefix.append(cp)
        if not (letter or number) and not re.match(r"\s", ch):
            punct.append(cp)
    return {
        "L": _bracketed(letters),
        "N": _bracketed(numbers),
        "PRE": _bracketed(prefix),
        "PUNCT": _bracketed(punct),
    }


def _bracketed(cps):
    """Coalesce consecutive code points into an escaped regex char class."""
    parts, start, prev = [], cps[0], cps[0]
    for cp in cps[1:]:
        if cp != prev + 1:
            parts.append(_range_str(start, prev))
            start = cp
        prev = cp
    parts.append(_range_str(start, prev))
    return "[" + "".join(parts) + "]"


def _range_str(a, b):
    if a == b:
        return f"\\U{a:08X}"
    return f"\\U{a:08X}-\\U{b:08X}"


class Qwen3Tokenizer:
    """Byte-level BPE with Qwen3's pre-tokenizer rules and special tokens."""

    _U = _unicode_classes()
    L, N, PRE, PUNCT = _U["L"], _U["N"], _U["PRE"], _U["PUNCT"]

    # Qwen3's pre-tokenizer (from the released tokenizer): contraction rules,
    # letter runs with an optional single non-letter prefix, one digit per
    # pre-token, punctuation runs that may absorb trailing newlines.
    PATTERN = (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        f"|{PRE}?{L}+"
        f"|{N}"
        f"| ?{PUNCT}+[\\r\\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+"
    )

    SPECIALS = (
        EOT,
        "<|im_start|>",
        "<|im_end|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|fim_pad|>",
        "<|repo_name|>",
        "<|file_sep|>",
    )

    BYTE_ENCODER = bytes_to_unicode()

    def __init__(self):
        self.base = sorted(self.BYTE_ENCODER.values())
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        self._special_ids = {}
        self._id_to_special = {}
        self._unicode_to_byte = {c: b for b, c in self.BYTE_ENCODER.items()}
        self._special_re = re.compile("|".join(map(re.escape, self.SPECIALS)))

    @classmethod
    def train(cls, text, vocab_size):
        """Learn merges on `text` so that len(vocab) == vocab_size."""
        tokenizer = cls()
        pretokens = tokenizer._pretokens(text)
        num_merges = vocab_size - len(tokenizer.base) - len(cls.SPECIALS)
        if num_merges < 1:
            raise ValueError(
                f"vocab_size={vocab_size} must exceed "
                f"{len(tokenizer.base) + len(cls.SPECIALS)} "
                "(256 base bytes + special tokens)"
            )
        tokenizer.merges = bpe_train(pretokens, num_merges)
        tokenizer.vocab = _compose_vocab(tokenizer.base, tokenizer.merges)
        tokenizer.inverse_vocab = {i: s for s, i in tokenizer.vocab.items()}
        for content in cls.SPECIALS:
            new_id = len(tokenizer.vocab)
            tokenizer.vocab[content] = new_id
            tokenizer.inverse_vocab[new_id] = content
            tokenizer._special_ids[content] = new_id
            tokenizer._id_to_special[new_id] = content
        return tokenizer

    def _pretokens(self, text):
        """Uppercased-byte symbols, one list per regex pre-token slice."""
        return [
            [self.BYTE_ENCODER[b] for b in chunk.encode("utf-8")]
            for chunk in re.findall(self.PATTERN, text)
        ]

    def _encode_symbols(self, text):
        ids = []
        for chunk in self._pretokens(text):
            ids.extend(self.vocab[s] for s in bpe_encode(chunk, self.merges))
        return ids

    def encode(self, text, allowed_special=()):
        """Encode `text`; special-token literals require allow-listing."""
        ids, pos = [], 0
        for match in self._special_re.finditer(text):
            content = match.group()
            if content not in allowed_special:
                raise ValueError(
                    f"Encountered text corresponding to disallowed "
                    f"special token {content!r}"
                )
            ids.extend(self._encode_symbols(text[pos : match.start()]))
            ids.append(self._special_ids[content])
            pos = match.end()
        ids.extend(self._encode_symbols(text[pos:]))
        return ids

    def decode(self, ids):
        pieces, byte_chunks = [], bytearray()

        def flush():
            if byte_chunks:
                pieces.append(bytes(byte_chunks).decode("utf-8", errors="replace"))
                byte_chunks.clear()

        for i in ids:
            if i in self._id_to_special:
                flush()
                pieces.append(self._id_to_special[i])
            else:
                for ch in self.inverse_vocab[i]:
                    byte_chunks.append(self._unicode_to_byte[ch])
        flush()
        return "".join(pieces)
