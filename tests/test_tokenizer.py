import pytest

from src.tokenizer import BPETokenizer, SimpleTokenizer


@pytest.fixture
def tokenizer():
    vocabulary = {
        "Hello": 0,
        ",": 1,
        "world": 2,
        "!": 3,
        "<|unk|>": 4,
    }
    return SimpleTokenizer(vocabulary)


def test_encode_known_words_and_punctuation(tokenizer):
    assert tokenizer.encode("Hello, world!") == [0, 1, 2, 3]


def test_encode_unknown_words_as_unknown_token(tokenizer):
    assert tokenizer.encode("Goodbye") == [4]


def test_encode_empty_text(tokenizer):
    assert tokenizer.encode("") == []


def test_decode_ids_and_punctuation(tokenizer):
    assert tokenizer.decode([0, 1, 2, 3]) == "Hello, world!"


def test_encode_decode_round_trip(tokenizer):
    text = "Hello, world!"
    assert tokenizer.decode(tokenizer.encode(text)) == text


@pytest.fixture
def bpe_tokenizer():
    corpus = "abababab! <|endoftext|> abababab!"
    return BPETokenizer(corpus, vocab_size=12)


def test_bpe_vocab_contains_base_and_special_tokens(bpe_tokenizer):
    expected_tokens = set("ab! ") | {"<|endoftext|>", "<|unk|>"}

    assert expected_tokens <= bpe_tokenizer.str_to_int.keys()


def test_bpe_vocab_mappings_are_inverses(bpe_tokenizer):
    for token, token_id in bpe_tokenizer.str_to_int.items():
        assert bpe_tokenizer.int_to_str[token_id] == token


def test_bpe_learns_merges_and_adds_merged_tokens_to_vocab(bpe_tokenizer):
    assert bpe_tokenizer.merge_rules

    for pair in bpe_tokenizer.merge_rules:
        assert "".join(pair) in bpe_tokenizer.str_to_int


def test_bpe_does_not_merge_endoftext_token(bpe_tokenizer):
    assert all(
        bpe_tokenizer.special_token not in pair for pair in bpe_tokenizer.merge_rules
    )


def test_bpe_merge_pair_merges_non_overlapping_occurrences(bpe_tokenizer):
    tokens = ["a", "a", "a", "a", "b"]

    assert bpe_tokenizer._merge_pair(tokens, ("a", "a")) == ["aa", "aa", "b"]


def test_bpe_encode_uses_learned_merges():
    tokenizer = BPETokenizer(
        "abababab", vocab_size=5
    )  # 5 because it will learn to merge "a" and "b" into "ab", and will include the special tokens "<|endoftext|>" and "<|unk|>"

    assert tokenizer.merge_rules == [("a", "b")]
    assert len(tokenizer.encode("abababab")) == 4


def test_bpe_encode_decode_round_trip(bpe_tokenizer):
    text = "abababab! <|endoftext|> abababab!"

    assert bpe_tokenizer.decode(bpe_tokenizer.encode(text)) == text


def test_bpe_unknown_character_decodes_as_unknown_token(bpe_tokenizer):
    assert "z" not in bpe_tokenizer.str_to_int

    encoded = bpe_tokenizer.encode("az")

    assert bpe_tokenizer.decode(encoded) == "a<|unk|>"


def test_bpe_encode_empty_text(bpe_tokenizer):
    assert bpe_tokenizer.encode("") == []


def test_bpe_decode_empty_ids(bpe_tokenizer):
    assert bpe_tokenizer.decode([]) == ""


def test_bpe_all_encoded_ids_are_in_vocabulary(bpe_tokenizer):
    encoded = bpe_tokenizer.encode("abab! z")

    assert all(token_id in bpe_tokenizer.int_to_str for token_id in encoded)


def test_bpe_allows_requested_vocab_smaller_than_base_vocab():
    tokenizer = BPETokenizer("abc", vocab_size=1)

    assert len(tokenizer.str_to_int) > tokenizer.vocab_size
    assert set("abc") <= tokenizer.str_to_int.keys()


def test_bpe_stops_when_no_mergeable_pairs_remain():
    tokenizer = BPETokenizer("a<|endoftext|>b", vocab_size=20)

    assert tokenizer.merge_rules == []
    assert len(tokenizer.str_to_int) < tokenizer.vocab_size
