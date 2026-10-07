import json
import re
from collections import Counter


def process_text(text):
    tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [item.strip() for item in tokens if item.strip()]
    return tokens


# Tokenize
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


class SentencePieceTokenizer:
    def __init__(self):
        self.str_to_int = {}
        self.int_to_str = {}
        self.merges = []

    def _process_text(self, text):
        words = text.split()
        return [tuple("\u2581" + word) for word in words]

    def _add_token(self, token):
        if token not in self.str_to_int:
            token_id = len(self.str_to_int)
            self.str_to_int[token] = token_id
            self.int_to_str[token_id] = token

    def _get_pair_counts(self, word_freqs):
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pairs[pair] += freq
        return pairs

    def _find_merge_candidate(self, word_freqs):
        pair_counts = self._get_pair_counts(word_freqs)
        if not pair_counts:
            return None
        return pair_counts.most_common(1)[0][0]

    def _merge_word(self, word, pair):
        first, second = pair
        merged = []
        i = 0
        while i < len(word):
            # If the adjacent pair matches, combine them and advance by 2
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                merged.append(first + second)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    def train(self, text, num_merges=100):
        self.str_to_int = {}
        self.int_to_str = {}
        word_tuples = self._process_text(text)
        word_freqs = Counter(word_tuples)

        # Save individual characters to the vocabulary before merging
        for word in word_freqs:
            for char in word:
                self._add_token(char)

        self._add_token("<|endoftext|>")
        self._add_token("<|unk|>")

        self.merges = []
        while len(self.merges) < num_merges:
            merge_candidate = self._find_merge_candidate(word_freqs)
            if not merge_candidate:
                break
            self.merges.append(merge_candidate)
            self._add_token(merge_candidate[0] + merge_candidate[1])

            # Perform the merges
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._merge_word(word, merge_candidate)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

    def encode(self, text):
        word_tuples = self._process_text(text)
        ids = []

        for word in word_tuples:
            # Fully apply all merges to this word
            for merge in self.merges:
                word = self._merge_word(word, merge)

            # Collect IDs for the finalized subwords
            for token in word:
                ids.append(self.str_to_int.get(token, self.str_to_int["<|unk|>"]))

        return ids

    def decode(self, ids):
        tokens = [self.int_to_str.get(i, "<|unk|>") for i in ids]
        text = "".join(tokens).replace("\u2581", " ").strip()
        return text


if __name__ == "__main__":  # pragma: no cover
    with open("SW_EpisodeIV_VI.json", "r") as f:
        data = json.load(f)

    lines = []
    for i in range(len(data)):
        lines.append(data[i]["Line"])

    # View the first 10 'Lines' in the dataset
    print(lines[0:10])

    preprocessed = process_text(" ".join(lines))

    # Build Vocab
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))

    # Using the simple tokenizer
    print("Using the simple tokenizer:")
    tokenizer = SimpleTokenizer(vocab)
    # Example of the tokenizer with all words in the vocabulary
    encoded = tokenizer.encode("Take the Falcon back to Tatooine.")
    decoded = tokenizer.decode(encoded)
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Example of the tokenizer with an unknown word
    encoded_unknown = tokenizer.encode("We must return to Coruscant.")
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f"Encoded with unknown: {encoded_unknown}")
    print(f"Decoded with unknown: {decoded_unknown}")

    # Using the Sentence Piece tokenizer
    print("Using the SentencePiece tokenizer:")
    sp_tokenizer = SentencePieceTokenizer()

    sp_tokenizer.train(" ".join(lines), num_merges=100)
    # Use the same examples as above to test the SentencePiece tokenizer
    encoded_sp = sp_tokenizer.encode("Take the Falcon back to Tatooine.")
    decoded_sp = sp_tokenizer.decode(encoded_sp)
    print(f"Encoded: {encoded_sp}")
    print(f"Decoded: {decoded_sp}")
    encoded_sp_unknown = sp_tokenizer.encode("We must return to Coruscant.")
    decoded_sp_unknown = sp_tokenizer.decode(encoded_sp_unknown)
    print(f"Encoded with unknown: {encoded_sp_unknown}")
    print(f"Decoded with unknown: {decoded_sp_unknown}")
