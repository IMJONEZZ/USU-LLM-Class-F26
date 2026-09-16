import json
import sys
from pathlib import Path

import pytest

from src.bpe_tokenizer import (
    DEFAULT_VOCAB_SIZE,
    BPETokenizer,
    get_pair_counts,
    load_corpus,
    main,
    merge_pair,
    text_to_byte_ids,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("ababab", [97, 98, 97, 98, 97, 98]),
        (" \t\n", [32, 9, 10]),
        ("é", [195, 169]),
        ("🙂", [240, 159, 153, 130]),
    ],
)
def test_text_to_byte_ids(text, expected):
    assert text_to_byte_ids(text) == expected


@pytest.mark.parametrize(
    ("token_ids", "expected"),
    [
        ([], {}),
        ([97], {}),
        ([97, 98], {(97, 98): 1}),
        ([97, 98, 97, 98, 97, 98], {(97, 98): 3, (98, 97): 2}),
        ([97, 97, 97, 97], {(97, 97): 3}),
        ([256, 256, 97], {(256, 256): 1, (256, 97): 1}),
    ],
)
def test_get_pair_counts(token_ids, expected):
    assert get_pair_counts(token_ids) == expected


@pytest.mark.parametrize(
    ("token_ids", "pair", "new_id", "expected"),
    [
        ([], (97, 98), 256, []),
        ([97], (97, 98), 256, [97]),
        ([97, 98], (97, 98), 256, [256]),
        ([97, 98, 97, 98, 97, 98], (97, 98), 256, [256, 256, 256]),
        ([97, 97, 97, 97], (97, 97), 256, [256, 256]),
        ([97, 97, 97], (97, 97), 256, [256, 97]),
        ([99, 97, 98, 100], (97, 98), 256, [99, 256, 100]),
        ([98, 97], (97, 98), 256, [98, 97]),
        ([256, 256, 256], (256, 256), 257, [257, 256]),
    ],
)
def test_merge_pair(token_ids, pair, new_id, expected):
    original = token_ids.copy()
    assert merge_pair(token_ids, pair, new_id) == expected
    assert token_ids == original


@pytest.fixture
def corpus_file(tmp_path):
    """A small complete corpus for predictable tests even without the real file."""
    path = tmp_path / "corpus.json"
    records = [
        {"Character": "LEIA", "Line": "  Hello, café!\n"},
        {"Character": "LUKE", "Line": "Last line🙂\t"},
    ]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_corpus_includes_every_field_and_preserves_whitespace(corpus_file):
    assert load_corpus(corpus_file) == "LEIA\n  Hello, café!\n\nLUKE\nLast line🙂\t"


def test_initial_vocabulary_contains_all_bytes():
    tokenizer = BPETokenizer()
    assert tokenizer.vocab == {i: bytes([i]) for i in range(256)}
    assert tokenizer.merges == {}


def test_train_learns_multiple_merges_in_order():
    tokenizer = BPETokenizer()
    ids = tokenizer.train("ababab", vocab_size=258)
    assert list(tokenizer.merges.items()) == [
        ((97, 98), 256),
        ((256, 256), 257),
    ]
    assert tokenizer.vocab[256] == b"ab"
    assert tokenizer.vocab[257] == b"abab"
    assert ids == [257, 256]
    assert len(tokenizer.vocab) == 258


def test_train_breaks_frequency_ties_by_token_ids():
    tokenizer = BPETokenizer()
    # Both pairs occur once; (97, 98) wins even though it appears second.
    assert tokenizer.train("bab", vocab_size=257) == [98, 256]
    assert tokenizer.merges == {(97, 98): 256}


def test_training_is_reproducible():
    first = BPETokenizer()
    second = BPETokenizer()
    assert first.train("bananas and bananas", 270) == second.train(
        "bananas and bananas", 270
    )
    assert list(first.merges.items()) == list(second.merges.items())
    assert first.vocab == second.vocab


def test_retraining_resets_previous_vocabulary_and_merges():
    tokenizer = BPETokenizer()
    tokenizer.train("ababab", vocab_size=258)
    assert tokenizer.train("cccc", vocab_size=257) == [256, 256]
    assert tokenizer.merges == {(99, 99): 256}
    assert tokenizer.vocab[256] == b"cc"
    assert len(tokenizer.vocab) == 257


@pytest.mark.parametrize(
    ("text", "vocab_size", "expected_ids", "expected_merges"),
    [
        ("", 512, [], 0),
        ("a", 512, [97], 0),
        ("ab", 512, [256], 1),
        ("abab", 256, [97, 98, 97, 98], 0),
    ],
)
def test_training_stops_at_target_or_when_no_pairs_remain(
    text, vocab_size, expected_ids, expected_merges
):
    tokenizer = BPETokenizer()
    assert tokenizer.train(text, vocab_size) == expected_ids
    assert len(tokenizer.merges) == expected_merges
    assert len(tokenizer.vocab) == 256 + expected_merges


def test_train_rejects_vocabulary_smaller_than_all_bytes():
    with pytest.raises(ValueError, match="at least 256"):
        BPETokenizer().train("Hello", vocab_size=255)


def test_training_preserves_complete_sample_corpus(corpus_file):
    text = load_corpus(corpus_file)
    tokenizer = BPETokenizer()
    ids = tokenizer.train(text, vocab_size=280)
    # Reconstruct bytes together: a token can contain part of a UTF-8 character.
    recovered = b"".join(tokenizer.vocab[token_id] for token_id in ids)
    assert recovered == text.encode("utf-8")
    assert len(ids) < len(text.encode("utf-8"))


def test_cli_trains_on_supplied_corpus(corpus_file, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["bpe_tokenizer.py", str(corpus_file), "--vocab-size", "257"]
    )
    main()
    output = capsys.readouterr().out
    assert "Vocabulary size: 257\n" in output
    assert "Learned merges: 1\n" in output
    assert f"Training characters: {len(load_corpus(corpus_file))}\n" in output


@pytest.fixture(scope="module")
def star_wars_corpus():
    """Use the complete real dataset locally; it is excluded from Git and CI."""
    path = Path(__file__).resolve().parents[1] / "SW_EpisodeIV_VI.json"
    if not path.exists():
        pytest.skip(
            "Full Star Wars dataset is not present; fixture-based tests still run"
        )
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    text = load_corpus(path)
    return records, text


@pytest.fixture(scope="module")
def trained_star_wars_tokenizer(star_wars_corpus):
    _, text = star_wars_corpus
    tokenizer = BPETokenizer()
    ids = tokenizer.train(text)
    return tokenizer, ids


def test_training_on_entire_star_wars_corpus(
    star_wars_corpus, trained_star_wars_tokenizer, tmp_path
):
    records, text = star_wars_corpus
    # Independently build the expected text from every field of every record.
    fields = []
    for record in records:
        fields.append(record["Character"])
        fields.append(record["Line"])
    assert text == "\n".join(fields)

    tokenizer, ids = trained_star_wars_tokenizer
    assert DEFAULT_VOCAB_SIZE == 4096
    assert len(tokenizer.vocab) == 4096
    assert len(tokenizer.merges) == 3840
    model_path = tmp_path / "star_wars_4096.bpe.json"
    tokenizer.save(model_path)
    restored = BPETokenizer.load(model_path)
    assert restored.vocab == tokenizer.vocab
    assert list(restored.merges.items()) == list(tokenizer.merges.items())
    assert restored.encode(text) == ids
    assert restored.decode(ids) == text
    assert len(ids) < len(text.encode("utf-8"))


@pytest.fixture
def trained_tokenizer():
    tokenizer = BPETokenizer()
    tokenizer.train("ababab", vocab_size=258)
    return tokenizer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("a", [97]),
        ("ababab", [257, 256]),
        ("abab", [257]),
        ("ab", [256]),
        ("bab", [98, 256]),
        ("abc", [256, 99]),
        ("é", [195, 169]),
        ("🙂", [240, 159, 153, 130]),
        (" \t\n", [32, 9, 10]),
    ],
)
def test_encode_reuses_learned_merges(trained_tokenizer, text, expected):
    assert trained_tokenizer.encode(text) == expected


def test_encode_preserves_the_trained_vocabulary(trained_tokenizer):
    vocab_before = trained_tokenizer.vocab.copy()
    merges_before = list(trained_tokenizer.merges.items())
    trained_tokenizer.encode("unseen text🙂 ababab")
    assert trained_tokenizer.vocab == vocab_before
    assert list(trained_tokenizer.merges.items()) == merges_before


def test_untrained_tokenizer_uses_byte_ids():
    tokenizer = BPETokenizer()
    assert tokenizer.encode("aé") == [97, 195, 169]
    assert tokenizer.decode([97, 195, 169]) == "aé"


@pytest.mark.parametrize(
    ("ids", "expected"),
    [
        ([], ""),
        ([97], "a"),
        ([257, 256], "ababab"),
        ([256, 99], "abc"),
        ([32, 9, 10], " \t\n"),
        ([195, 169], "é"),
        ([240, 159, 153, 130], "🙂"),
    ],
)
def test_decode_reconstructs_bytes_before_text(trained_tokenizer, ids, expected):
    assert trained_tokenizer.decode(ids) == expected


@pytest.mark.parametrize("invalid_id", [-1, 9999])
def test_decode_rejects_unknown_ids(trained_tokenizer, invalid_id):
    with pytest.raises(KeyError):
        trained_tokenizer.decode([invalid_id])


def test_decode_rejects_incomplete_utf8(trained_tokenizer):
    with pytest.raises(UnicodeDecodeError):
        trained_tokenizer.decode([195])


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  Hello\tthere!\n",
        "newwordneverinthecorpus",
        "café🙂",
        "你好 دنیا",
        "e\u0301",  # Preserve the combining accent exactly, without normalization.
        "<|unk|> <|endoftext|>",  # These strings are ordinary text here.
    ],
)
def test_round_trip_of_unseen_text_with_small_fixture(trained_tokenizer, text):
    assert trained_tokenizer.decode(trained_tokenizer.encode(text)) == text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  Hello\tthere!\n",
        "newwordneverinthecorpus",
        "café🙂",
        "你好 دنیا",
        "e\u0301",
        "<|unk|> <|endoftext|>",
    ],
)
def test_round_trip_after_full_corpus_training(trained_star_wars_tokenizer, text):
    tokenizer, _ = trained_star_wars_tokenizer
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_cli_encodes_and_decodes_new_text(corpus_file, monkeypatch, capsys):
    sample = "Hello🙂\n"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bpe_tokenizer.py",
            str(corpus_file),
            "--vocab-size",
            "257",
            "--text",
            sample,
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "Vocabulary size: 257\n" in output
    assert "Encoded IDs: [" in output
    assert f"Decoded text: {sample!r}\n" in output


@pytest.fixture
def saved_tokenizer(trained_tokenizer, tmp_path):
    path = tmp_path / "tokenizer.json"
    trained_tokenizer.save(path)
    return path


def test_save_records_bytes_and_ordered_merges(saved_tokenizer):
    model = json.loads(saved_tokenizer.read_text(encoding="utf-8"))
    assert model["format_version"] == 1
    assert model["vocab"][195] == "c3"  # Not a complete UTF-8 character by itself.
    assert model["vocab"][256:] == ["6162", "61626162"]
    assert model["merges"] == [[97, 98, 256], [256, 256, 257]]


def test_save_creates_parent_directories(trained_tokenizer, tmp_path):
    path = tmp_path / "models" / "nested" / "tokenizer.json"
    trained_tokenizer.save(path)
    assert BPETokenizer.load(path).encode("ababab") == [257, 256]


def test_save_load_preserves_all_state_and_encoding(trained_tokenizer, saved_tokenizer):
    restored = BPETokenizer.load(saved_tokenizer)
    assert restored.vocab == trained_tokenizer.vocab
    assert list(restored.merges.items()) == list(trained_tokenizer.merges.items())
    for text in ["ababab", "bab", "", "  café🙂\n你好 دنیا\t"]:
        assert restored.encode(text) == trained_tokenizer.encode(text)
        assert restored.decode(restored.encode(text)) == text


def test_save_load_untrained_byte_vocabulary(tmp_path):
    path = tmp_path / "bytes.json"
    BPETokenizer().save(path)
    restored = BPETokenizer.load(path)
    assert len(restored.vocab) == 256
    assert restored.merges == {}
    assert restored.encode("é") == [195, 169]
    assert restored.decode([195, 169]) == "é"


def test_loaded_tokenizer_is_independent(trained_tokenizer, saved_tokenizer):
    restored = BPETokenizer.load(saved_tokenizer)
    restored.train("cccc", vocab_size=257)
    assert trained_tokenizer.encode("ababab") == [257, 256]
    assert restored.encode("cccc") == [256, 256]


@pytest.mark.parametrize(
    ("model", "error_type"),
    [
        ([], ValueError),
        ({"format_version": 2}, ValueError),
        ({"format_version": 1}, TypeError),
        ({"format_version": 1, "vocab": {}, "merges": []}, TypeError),
        ({"format_version": 1, "vocab": [], "merges": {}}, TypeError),
        ({"format_version": 1, "vocab": ["not hex"], "merges": []}, ValueError),
        ({"format_version": 1, "vocab": [123], "merges": []}, ValueError),
        ({"format_version": 1, "vocab": [], "merges": []}, ValueError),
    ],
)
def test_load_rejects_invalid_model_structure(tmp_path, model, error_type):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(error_type):
        BPETokenizer.load(path)


@pytest.mark.parametrize(
    "rules",
    [
        ["bad rule"],
        [[97, 98]],
        [[97, 98, "256"]],
        [[True, 98, 256]],
        [[97, 98, 257]],  # The first merged ID must be 256.
        [[999, 98, 256]],  # A dependency is missing.
        [[97, -1, 256]],
        [[256, 256, 257], [97, 98, 256]],  # Reversed training order.
        [[97, 98, 256], [97, 98, 257]],  # A pair is defined twice.
    ],
)
def test_load_rejects_invalid_merge_rules(saved_tokenizer, rules):
    model = json.loads(saved_tokenizer.read_text(encoding="utf-8"))
    model["merges"] = rules
    saved_tokenizer.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="merge|Merge"):
        BPETokenizer.load(saved_tokenizer)


@pytest.mark.parametrize("token_id", [0, 256])
def test_load_rejects_mismatched_vocabulary(saved_tokenizer, token_id):
    model = json.loads(saved_tokenizer.read_text(encoding="utf-8"))
    model["vocab"][token_id] = "ff"
    saved_tokenizer.write_text(json.dumps(model), encoding="utf-8")
    with pytest.raises(ValueError, match="vocabulary does not match"):
        BPETokenizer.load(saved_tokenizer)


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        BPETokenizer.load(tmp_path / "missing.json")


def test_cli_saves_trained_model(corpus_file, tmp_path, monkeypatch, capsys):
    path = tmp_path / "trained.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bpe_tokenizer.py",
            str(corpus_file),
            "--vocab-size",
            "257",
            "--save",
            str(path),
        ],
    )
    main()
    assert f"Saved tokenizer: {path}" in capsys.readouterr().out
    assert len(BPETokenizer.load(path).vocab) == 257


def test_cli_loads_without_corpus_or_training(saved_tokenizer, monkeypatch, capsys):
    def unexpected_training(*args, **kwargs):
        pytest.fail("Loading a saved model must not read the corpus or retrain")

    monkeypatch.setattr("src.bpe_tokenizer.load_corpus", unexpected_training)
    monkeypatch.setattr(BPETokenizer, "train", unexpected_training)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bpe_tokenizer.py", "--load", str(saved_tokenizer), "--text", "ababab"],
    )
    main()
    output = capsys.readouterr().out
    assert "Vocabulary size: 258\n" in output
    assert "Encoded IDs: [257, 256]\n" in output
    assert "Decoded text: 'ababab'\n" in output
    assert "Training characters:" not in output


@pytest.mark.parametrize("training_args", [["corpus.json"], ["--vocab-size", "4096"]])
def test_cli_rejects_training_arguments_when_loading(
    training_args, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys, "argv", ["bpe_tokenizer.py", "--load", "model.json", *training_args]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--load cannot be combined" in capsys.readouterr().err


@pytest.fixture(scope="module")
def shared_tokenizer():
    """The committed model must be usable in CI without the training corpus."""
    path = Path(__file__).resolve().parents[1] / "models" / "star_wars_4096.bpe.json"
    return BPETokenizer.load(path)


def test_shared_model_has_expected_vocabulary(shared_tokenizer):
    assert len(shared_tokenizer.vocab) == 4096
    assert len(shared_tokenizer.merges) == 3840


@pytest.mark.parametrize(
    "text", ["", "newwordneverinthecorpus", "  Hello\tthere!\n", "café🙂 你好 دنیا"]
)
def test_shared_model_preserves_new_text(shared_tokenizer, text):
    assert shared_tokenizer.decode(shared_tokenizer.encode(text)) == text
