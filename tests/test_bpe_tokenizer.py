import json

import pytest

from src.bpe_tokenizer import (
    END_OF_TEXT_ID,
    MINIMUM_VOCAB_SIZE,
    UNKNOWN_ID,
    BytePairTokenizer,
    count_pairs,
    main,
    merge_pair,
)


def test_count_pairs_counts_adjacent_pairs():
    counts = count_pairs([1, 2, 1, 2])

    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1


def test_merge_pair_replaces_non_overlapping_pairs():
    merged = merge_pair([1, 1, 1], (1, 1), 258)

    assert merged == [258, 1]


def test_vocab_size_cannot_be_too_small():
    with pytest.raises(ValueError, match="must be at least 258"):
        BytePairTokenizer(target_vocab_size=257)


def test_initial_vocabulary_contains_bytes_and_special_tokens():
    tokenizer = BytePairTokenizer()

    assert tokenizer.vocabulary_size == MINIMUM_VOCAB_SIZE
    assert tokenizer.decode([END_OF_TEXT_ID]) == "<|endoftext|>"
    assert tokenizer.decode([UNKNOWN_ID]) == "<|unk|>"


def test_training_learns_frequent_pairs():
    tokenizer = BytePairTokenizer(target_vocab_size=260)

    tokenizer.train("abababab")

    assert tokenizer.merges[(ord("a"), ord("b"))] == 258
    assert tokenizer.vocabulary_size == 260


def test_bpe_compresses_repeated_text():
    text = "hello hello hello"
    tokenizer = BytePairTokenizer(target_vocab_size=270)

    tokenizer.train(text)
    encoded = tokenizer.encode(text)

    assert len(encoded) < len(text.encode("utf-8"))


def test_encode_decode_preserves_original_text():
    text = "They've arrived!  We'll begin."
    tokenizer = BytePairTokenizer(target_vocab_size=280)

    tokenizer.train(text * 5)
    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded) == text


def test_unseen_unicode_text_does_not_need_unknown_token():
    tokenizer = BytePairTokenizer(target_vocab_size=260)
    tokenizer.train("hello hello")
    unseen_text = "नमस्ते 😊"

    encoded = tokenizer.encode(unseen_text)

    assert UNKNOWN_ID not in encoded
    assert tokenizer.decode(encoded) == unseen_text


def test_training_stops_when_no_pair_repeats():
    tokenizer = BytePairTokenizer(target_vocab_size=512)

    tokenizer.train("ab")

    assert tokenizer.merges == {}
    assert tokenizer.vocabulary_size == MINIMUM_VOCAB_SIZE


def test_decode_rejects_invalid_token_id():
    tokenizer = BytePairTokenizer()

    with pytest.raises(ValueError, match="Unknown token ID: 999"):
        tokenizer.decode([999])


def test_main_runs_with_dialogue_dataset(tmp_path, capsys):
    dataset = [
        {"Character": "THREEPIO", "Line": "We're ready!"},
        {"Character": "LUKE", "Line": "Let's go!"},
    ]
    dataset_path = tmp_path / "dialogue.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    main(dataset_path)

    output = capsys.readouterr().out
    assert "Vocabulary size:" in output
    assert "Learned merges:" in output
    assert "BPE tokens:" in output
    assert "Decoded: We're ready!" in output
