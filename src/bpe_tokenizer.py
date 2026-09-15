import json
import re
from collections import Counter

DATA_PATH = "data/SW_EpisodeIV_VI.json"
END_OF_WORD = "</w>"


def load_dialogue(path):
    """Load the dataset and join all dialogue lines into one text blob."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    lines = [entry["Line"] for entry in data]
    return "\n".join(lines)


def get_word_counts(text):
    """
    Split text into words and count frequency of each word.
    Each word is represented as a tuple of characters plus an end-of-word marker,
    e.g. "cat" -> ('c', 'a', 't', '</w>')
    """
    words = re.findall(r"\S+", text)
    word_counts = Counter(words)
    return {tuple(word) + (END_OF_WORD,): count for word, count in word_counts.items()}


def get_pair_counts(word_counts):
    """Count frequency of every adjacent symbol pair across all words."""
    pair_counts = Counter()
    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair_counts[(word[i], word[i + 1])] += count
    return pair_counts


def merge_pair(pair, word_counts):
    """Merge every occurrence of `pair` into a single symbol across all words."""
    new_word_counts = {}
    first, second = pair
    merged = first + second

    for word, count in word_counts.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word_counts[tuple(new_word)] = count

    return new_word_counts


def train_bpe(text, num_merges=300):
    """
    Learn BPE merge rules from text.
    Returns the ordered list of merges and the final vocabulary set.

    Important: the base character set (every individual character seen in
    training) is always included in the vocabulary, even if a character gets
    merged into a larger symbol every time it appears. Without this, a
    character could vanish from the vocab entirely and break encoding for
    any unseen word containing it.
    """
    word_counts = get_word_counts(text)

    # Base vocab: every individual character seen in training. These are the
    # ultimate fallback for encoding any word, seen or unseen.
    vocab_symbols = set()
    for word in word_counts:
        vocab_symbols.update(word)

    merges = []
    for _ in range(num_merges):
        pair_counts = get_pair_counts(word_counts)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        word_counts = merge_pair(best_pair, word_counts)
        merges.append(best_pair)
        merged_symbol = best_pair[0] + best_pair[1]
        vocab_symbols.add(merged_symbol)

    return merges, vocab_symbols


class BPETokenizer:
    def __init__(self, merges, vocab_symbols):
        """
        merges: ordered list of (first, second) tuples learned during training
        vocab_symbols: set of all known subword symbols
        """
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

        all_tokens = sorted(vocab_symbols) + ["<|endoftext|>", "<|unk|>"]
        self.str_to_int = {token: i for i, token in enumerate(all_tokens)}
        self.int_to_str = {i: token for token, i in self.str_to_int.items()}

    def _bpe_word(self, word):
        """Apply learned merges to a single word (tuple of characters + end marker)."""
        symbols = list(word)

        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find the pair with the lowest merge rank (i.e. learned earliest / most useful)
            ranked_pairs = [
                (self.merge_ranks[p], p) for p in pairs if p in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs, key=lambda x: x[0])

            first, second = best_pair
            merged = first + second
            new_symbols = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == first
                    and symbols[i + 1] == second
                ):
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text):
        words = re.findall(r"\S+", text)
        ids = []
        for word in words:
            char_word = tuple(word) + (END_OF_WORD,)
            subwords = self._bpe_word(char_word)
            for sw in subwords:
                token_id = self.str_to_int.get(sw, self.str_to_int["<|unk|>"])
                ids.append(token_id)
        return ids

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = "".join(tokens)
        text = text.replace(END_OF_WORD, " ")
        return text.strip()


if __name__ == "__main__":  # pragma: no cover
    raw_text = load_dialogue(DATA_PATH)
    merges, vocab_symbols = train_bpe(raw_text, num_merges=300)
    print(f"Learned {len(merges)} merges")
    print(f"Vocab size: {len(vocab_symbols) + 2}")  # +2 for special tokens

    tokenizer = BPETokenizer(merges, vocab_symbols)
    sample_text = "Did you hear that? We're doomed!"
    ids = tokenizer.encode(sample_text)
    print(ids)
    print(tokenizer.decode(ids))
