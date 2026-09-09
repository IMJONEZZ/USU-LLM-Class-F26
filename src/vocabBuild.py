from collections import Counter


def initial_wordpiece_split(token):
    """Break token into individual characters. First character is left as-is, subsequent characters are prefixed with '##'."""

    if len(token) == 1 or not any(char.isalnum() for char in token):
        return [token]

    return [token[0]] + ["##" + char for char in token[1:]]


def compute_pair_stats(splits, token_freqs):
    """Count how often individual WordPiece pieces and adjacent pairs occur.
    These counts are used to determine which two pieces should be merged
    next during vocabulary training."""

    pair_freqs = Counter()
    piece_freqs = Counter()

    for token, split in splits.items():
        frequency = token_freqs[token]

        for piece in split:
            piece_freqs[piece] += frequency

        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += frequency

    return pair_freqs, piece_freqs


def get_best_pair(splits, token_freqs):
    """Find the pair with the highest WordPiece score."""

    pair_freqs, piece_freqs = compute_pair_stats(splits, token_freqs)

    best_pair = None
    best_score = -1.0

    for pair, pair_frequency in pair_freqs.items():
        left, right = pair

        score = pair_frequency / (piece_freqs[left] * piece_freqs[right])

        if score > best_score:
            best_score = score
            best_pair = pair

    return best_pair


def merge_tokens(left, right):
    """Merge two WordPiece pieces."""

    right = right.removeprefix("##")

    return left + right


def merge_pair(pair, splits):
    """Updates the current representation of every word after WordPiece
    decides that two pieces should become a single vocabulary token."""

    left, right = pair
    merged = merge_tokens(left, right)

    new_splits = {}

    for token, split in splits.items():
        new_split = []
        i = 0

        while i < len(split):
            if i < len(split) - 1 and split[i] == left and split[i + 1] == right:
                new_split.append(merged)
                i += 2

            else:
                new_split.append(split[i])
                i += 1

        new_splits[token] = new_split

    return new_splits, merged


def build_wordpiece_vocab(preprocessed, target_vocab_size=1000):
    """Train the complete WordPiece vocabulary.

    Training begins by counting how often each input token occurs and
    splitting those tokens into character-level WordPiece pieces.

    It then repeatedly:

        1. Counts adjacent piece pairs.
        2. Scores those pairs.
        3. Selects the highest-scoring pair.
        4. Merges that pair into a new subword token.
        5. Adds the new token to the vocabulary.

    This process continues until the target vocabulary size is reached
    or there are no more pairs that can be merged.

    Finally, every vocabulary token is assigned a unique integer ID."""

    token_freqs = Counter(preprocessed)

    # Split each unique token into its starting WordPiece pieces.
    splits = {}

    for token in token_freqs:
        splits[token] = initial_wordpiece_split(token)

    # Collect every unique WordPiece piece into the starting vocabulary.
    vocab = set()

    for split in splits.values():
        for piece in split:
            vocab.add(piece)

    vocab.update(["<|endoftext|>", "<|unk|>"])

    while len(vocab) < target_vocab_size:
        best_pair = get_best_pair(splits, token_freqs)

        if best_pair is None:
            break

        # Reset splits to reflect the new merged token.
        splits, new_token = merge_pair(best_pair, splits)

        vocab.add(new_token)

    all_tokens = sorted(vocab)

    vocab_dictionary = {}

    for integer, token in enumerate(all_tokens):
        vocab_dictionary[token] = integer

    return vocab_dictionary
