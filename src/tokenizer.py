import json
import re
from collections import defaultdict

STARWARS_PATH = "./data/SW_EpisodeIV_VI.json"


def load_starwars_data():
    with open(STARWARS_PATH, "r") as f:
        starwars_data = json.load(f)
    preprocessed = []
    for character_line in starwars_data:
        pieces = re.split(r'([,.:;?_!"()\']|--|\s)', character_line["Line"])
        preprocessed.extend(piece.strip() for piece in pieces if piece.strip())
    return preprocessed


class BPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.str_to_int = {}
        self.int_to_str = {}

    def compute_word_freqs(self, preprocessed):
        word_freqs = defaultdict(int)
        for word in preprocessed:
            word_freqs[word] += 1
        return word_freqs

    def compute_splits(self, word_freqs):
        return {word: ["Ġ"] + list(word) for word in word_freqs}

    def compute_pair_freqs(self, splits, word_freqs):
        pair_freqs = {}
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                if pair not in pair_freqs:
                    pair_freqs[pair] = 0
                pair_freqs[pair] += freq
        return pair_freqs

    def merge_pair(self, a, b, splits, word_freqs):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def compute_alphabet(self, word_freqs):
        alphabet = set()
        for word in word_freqs:
            for char in word:
                alphabet.add(char)
        return sorted(alphabet)

    def train(self, preprocessed):
        word_freqs = self.compute_word_freqs(preprocessed)
        alphabet = self.compute_alphabet(word_freqs)
        base_tokens = alphabet + ["Ġ"]
        for i, token in enumerate(base_tokens):
            self.str_to_int[token] = i
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}

        splits = self.compute_splits(word_freqs)
        merges = {}
        needed_merges = self.vocab_size - len(self.str_to_int)
        for _ in range(needed_merges):
            pair_freqs = self.compute_pair_freqs(splits, word_freqs)
            if not pair_freqs:
                break
            best_pair = max(pair_freqs, key=pair_freqs.get)
            splits = self.merge_pair(best_pair[0], best_pair[1], splits, word_freqs)
            new_token = best_pair[0] + best_pair[1]
            new_id = len(self.str_to_int)
            self.str_to_int[new_token] = new_id
            self.int_to_str[new_id] = new_token
            merges[best_pair] = new_token

        self.splits = splits
        self.merges = merges

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = []
        for word in preprocessed:
            for piece in self.splits[word]:
                ids.append(self.str_to_int[piece])
        return ids

    def decode(self, ids):
        text = "".join(self.int_to_str[i] for i in ids)
        text = text.replace("Ġ", " ").strip()
        return re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
