from collections import Counter
from itertools import pairwise
from pathlib import Path

import regex

PRETOKENIZATION_PATTERN = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPE_Algorithm:
    def __init__(self, training_text, merge_size):
        self.training_text = training_text.lower()
        self.merge_size = merge_size
        self.bpe_merges: list[str] = []

    def pretokenize(self) -> list[tuple[str, int]]:
        """Return GPT-2-style pre-token pieces paired with their frequencies."""
        return list(
            Counter(PRETOKENIZATION_PATTERN.findall(self.training_text)).items()
        )

    def base_vocab(
        self, pretokenized: list[tuple[str, int]]
    ) -> list[tuple[list[str], int]]:
        """Build character vocab entries and add unique characters to ``bpe_merges``."""
        vocabulary = []
        for word, count in pretokenized:
            for character in word:
                if character not in self.bpe_merges:
                    self.bpe_merges.append(character)
            vocabulary.append((list(word), count))

        return vocabulary

    def merge_pairs(
        self, text: list[tuple[list[str], int]]
    ) -> list[tuple[list[str], int]]:
        """Merge the most frequent adjacent pair, weighted by word frequency."""
        pair_counts: Counter[tuple[str, str]] = Counter()
        for characters, count in text:
            for pair in pairwise(characters):
                pair_counts[pair] += count

        if not pair_counts:
            return text

        pair_to_merge = pair_counts.most_common(1)[0][0]
        merged_token = "".join(pair_to_merge)
        self.bpe_merges.append(merged_token)

        updated_text = []
        for characters, count in text:
            merged_characters = []
            index = 0
            while index < len(characters):
                if (
                    index + 1 < len(characters)
                    and (characters[index], characters[index + 1]) == pair_to_merge
                ):
                    merged_characters.append(merged_token)
                    index += 2
                else:
                    merged_characters.append(characters[index])
                    index += 1
            updated_text.append((merged_characters, count))

        return updated_text

    def mergify(self) -> None:
        """Build the base vocabulary and merge pairs until ``merge_size`` is reached."""
        vocabulary = self.base_vocab(self.pretokenize())

        while len(self.bpe_merges) < self.merge_size:
            merge_count = len(self.bpe_merges)
            vocabulary = self.merge_pairs(vocabulary)

            if len(self.bpe_merges) == merge_count:
                break


class BSE_Vocabulary:
    """Assign stable integer IDs to learned BPE tokens."""

    unknown_token = "<|unk|>"

    def __init__(self, bpe_merges: list[str]):
        tokens = [token for token in bpe_merges if token != self.unknown_token]
        tokens.append(self.unknown_token)

        self.str2id = {token: token_id for token_id, token in enumerate(tokens)}
        self.id2str = {token_id: token for token, token_id in self.str2id.items()}


class BSE_Encoder:
    """Encode and decode words with a greedy BPE vocabulary."""

    def __init__(self, str2id: dict[str, int]):
        self.str2id = str2id
        self.id2str = {token_id: token for token, token_id in str2id.items()}
        self.unknown_token = BSE_Vocabulary.unknown_token
        self.unknown_id = self.str2id[self.unknown_token]
        self.bpe_merges = [
            token for token in self.str2id if token != self.unknown_token
        ]
        self._tokens_by_length = sorted(self.bpe_merges, key=len, reverse=True)

    @staticmethod
    def get_pairs(word: str) -> list[tuple[str, str]]:
        """Return adjacent character pairs from a word."""
        word = word.lower()
        return list(pairwise(word))

    def bpe(self, word: str) -> list[str]:
        """Split a word by greedily choosing the longest matching BPE token."""
        word = word.lower()
        tokens = []
        position = 0

        while position < len(word):
            match = next(
                (
                    token
                    for token in self._tokens_by_length
                    if word.startswith(token, position)
                ),
                None,
            )
            if match is None:
                match = word[position]

            tokens.append(match)
            position += len(match)

        return tokens

    def encode(self, word: str) -> list[int]:
        """Encode a word into BPE token IDs."""
        return [self.str2id.get(token, self.unknown_id) for token in self.bpe(word)]

    def decode(self, token_ids: list[int]) -> str:
        """Reconstruct text from BPE token IDs."""
        return "".join(
            self.id2str.get(token_id, self.unknown_token) for token_id in token_ids
        )


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "sports.txt"
    training_text = data_path.read_text(encoding="utf-8")

    algorithm = BPE_Algorithm(training_text=training_text, merge_size=3000)
    algorithm.mergify()
    vocabulary = BSE_Vocabulary(algorithm.bpe_merges)
    encoder = BSE_Encoder(vocabulary.str2id)

    sentence = "Which basketball team do you think will win the championship this year?"
    token_ids = encoder.encode(sentence)
    tokens = [vocabulary.id2str[token_id] for token_id in token_ids]

    print(f"Encoded: {token_ids}\n")
    print(f"Tokens: {tokens}\n")
    print(f"Decoded: {encoder.decode(token_ids)}")
