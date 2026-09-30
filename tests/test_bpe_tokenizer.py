"""Tests for the byte-level BPE tokenizer.

These are organised around the four claims made in the plan of action, since
those are what "better than SimpleTokenizer" actually means here:

  1. it reproduces the BPE algorithm correctly (the HuggingFace worked example)
  2. no input is ever unknown, so nothing is silently destroyed
  3. encode -> decode returns the input exactly
  4. vocabulary size is something we choose, not something the corpus decides

The rest cover the supporting machinery: saving and loading, special tokens,
determinism, and the error cases.
"""

import json

import pytest

from src.bpe_tokenizer import (
    BYTE_VOCAB_SIZE,
    END_OF_TEXT,
    SPECIAL_TOKENS,
    UNKNOWN,
    BPETokenizer,
    _count_pairs,
    _merge,
    extract_strings,
    load_corpus,
    main,
    measure,
)

# The example from HuggingFace's tokenizer summary: hug x10, pug x5, pun x12,
# bun x4, hugs x5. Their documented merge order is ug, then un, then hug.
HF_EXAMPLE = " ".join(
    ["hug"] * 10 + ["pug"] * 5 + ["pun"] * 12 + ["bun"] * 4 + ["hugs"] * 5
)

# Small stand-in for the real dataset so tests never need the gitignored file.
SAMPLE_CORPUS = (
    "Luke, use the Force. The Force is strong with this one. "
    "Leia said the Force would guide him, and the Force did. "
    "Han shot first, then Han ran, and running was all Han did."
)


@pytest.fixture
def hf_tokenizer():
    """Tokenizer trained on HuggingFace's worked example, 4 merges only."""
    return BPETokenizer().train(HF_EXAMPLE, vocab_size=BYTE_VOCAB_SIZE + 4 + 2)


@pytest.fixture
def trained():
    """Tokenizer trained on the sample corpus with a realistic vocab size."""
    return BPETokenizer().train(SAMPLE_CORPUS, vocab_size=400)


@pytest.fixture
def dataset_dir(tmp_path, monkeypatch):
    """A temp directory holding a fake SW_EpisodeIV_VI.json, cwd'd into."""
    payload = {
        "title": "A New Hope",
        "scenes": [
            {"speaker": "LUKE", "dialogue": SAMPLE_CORPUS},
            {"speaker": "LEIA", "dialogue": SAMPLE_CORPUS},
        ],
        "episode": 4,
        "released": True,
    }
    (tmp_path / "SW_EpisodeIV_VI.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- 1. the algorithm itself -------------------------------------------------


def test_reproduces_huggingface_worked_example(hf_tokenizer):
    """The documented merges for hug/pug/pun/bun must all be learned."""
    learned = {hf_tokenizer.vocab[new_id] for new_id in hf_tokenizer.merges.values()}
    assert b"ug" in learned
    assert b"un" in learned
    assert b"hug" in learned


def test_most_frequent_pair_is_merged_first(hf_tokenizer):
    """BPE merges on raw frequency, so 'ug' (20 occurrences) comes first."""
    first_pair = next(iter(hf_tokenizer.merges))
    assert hf_tokenizer.vocab[hf_tokenizer.merges[first_pair]] == b"ug"


def test_merged_token_is_concatenation_of_its_parts(trained):
    """Every learned token must equal the two tokens it was built from."""
    for (first, second), new_id in trained.merges.items():
        assert trained.vocab[new_id] == trained.vocab[first] + trained.vocab[second]


def test_count_pairs_weights_by_word_frequency():
    """A pair inside a word seen 10 times counts 10 times, not once."""
    counts = _count_pairs({(1, 2): 10, (2, 3): 1})
    assert counts[(1, 2)] == 10
    assert counts[(2, 3)] == 1


def test_merge_replaces_every_occurrence():
    assert _merge((1, 2, 1, 2, 3), (1, 2), 99) == (99, 99, 3)


def test_merge_leaves_non_matching_symbols_alone():
    assert _merge((1, 2, 3), (7, 8), 99) == (1, 2, 3)


def test_training_is_deterministic():
    """Same text and size must give the same merges, run to run."""
    first = BPETokenizer().train(SAMPLE_CORPUS, vocab_size=400)
    second = BPETokenizer().train(SAMPLE_CORPUS, vocab_size=400)
    assert first.merges == second.merges


# --- 2. nothing is ever unknown ---------------------------------------------


def test_unseen_word_encodes_without_unknown_token(trained):
    """The word SimpleTokenizer destroys must survive here."""
    ids = trained.encode("Chewbacca")
    assert ids
    assert trained.special_to_id[UNKNOWN] not in ids
    assert trained.decode(ids) == "Chewbacca"


@pytest.mark.parametrize(
    "text",
    [
        "Chewbacca",
        "ANAKIN",
        "quixotic zephyr",
        "héllo",
        "🌍 emoji",
        "1138",
    ],
)
def test_never_emits_unknown_for_unseen_input(trained, text):
    assert trained.special_to_id[UNKNOWN] not in trained.encode(text)


def test_untrained_tokenizer_still_encodes_everything():
    """With no merges at all it degrades to raw bytes, never to <|unk|>."""
    tokenizer = BPETokenizer()
    assert tokenizer.decode(tokenizer.encode("anything at all")) == "anything at all"


# --- 3. exact round trips ----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Luke, use the Force.",
        "Chewbacca",
        "",
        "   ",
        "  spaced   out  ",
        "line one\nline two\n",
        "tab\tseparated",
        "héllo wörld",
        "🌍🚀 emoji run",
        "punctuation!?;:-- ()'\"",
        "1138 and 06 and 47",
        "trailing space ",
    ],
)
def test_round_trip_is_exact(trained, text):
    assert trained.decode(trained.encode(text)) == text


def test_round_trip_preserves_repeated_whitespace(trained):
    """SimpleTokenizer's regex decode collapsed these; this one must not."""
    text = "a  b   c"
    assert trained.decode(trained.encode(text)) == text


def test_empty_text_encodes_to_nothing(trained):
    assert trained.encode("") == []
    assert trained.decode([]) == ""


# --- 4. vocabulary size is chosen, not discovered ---------------------------


def test_vocab_size_target_is_reached_when_the_corpus_allows():
    tokenizer = BPETokenizer().train(SAMPLE_CORPUS, vocab_size=300)
    assert tokenizer.vocab_size == 300


def test_vocab_size_is_an_upper_bound_not_a_guarantee(trained):
    """A corpus can run out of pairs before the target is reached.

    The sample corpus only supports about 80 merges, so asking for 400 gives
    fewer. That is deliberate -- padding the vocabulary with junk tokens to hit
    a number would be worse than stopping.
    """
    assert trained.vocab_size <= 400


def test_base_vocabulary_covers_every_byte():
    tokenizer = BPETokenizer()
    assert tokenizer.vocab_size == BYTE_VOCAB_SIZE + len(SPECIAL_TOKENS)
    for byte_value in range(BYTE_VOCAB_SIZE):
        assert tokenizer.vocab[byte_value] == bytes([byte_value])


def test_larger_vocab_gives_shorter_sequences():
    """More merges means fewer tokens for the same text."""
    small = BPETokenizer().train(SAMPLE_CORPUS, vocab_size=300)
    large = BPETokenizer().train(SAMPLE_CORPUS, vocab_size=400)
    assert len(large.encode(SAMPLE_CORPUS)) < len(small.encode(SAMPLE_CORPUS))


def test_vocab_size_below_the_byte_floor_is_rejected():
    with pytest.raises(ValueError, match="at least"):
        BPETokenizer().train(SAMPLE_CORPUS, vocab_size=10)


def test_training_stops_early_when_nothing_is_left_to_merge():
    """A tiny corpus runs out of pairs before the target size is reached."""
    tokenizer = BPETokenizer().train("ab ab", vocab_size=900)
    assert len(tokenizer.merges) < 900 - BYTE_VOCAB_SIZE - len(SPECIAL_TOKENS)


def test_verbose_training_reports_progress(capsys):
    BPETokenizer().train(SAMPLE_CORPUS, vocab_size=260, verbose=True)
    assert "merge 1/" in capsys.readouterr().out


# --- special tokens ----------------------------------------------------------


def test_special_token_encodes_as_a_single_id(trained):
    assert trained.encode(END_OF_TEXT) == [trained.special_to_id[END_OF_TEXT]]


def test_special_token_survives_inside_surrounding_text(trained):
    text = f"before{END_OF_TEXT}after"
    ids = trained.encode(text)
    assert trained.special_to_id[END_OF_TEXT] in ids
    assert trained.decode(ids) == text


def test_special_tokens_sit_above_the_learned_merges(trained):
    """Their IDs must not collide with byte or merge IDs."""
    merge_ids = set(trained.merges.values())
    for token_id in trained.special_to_id.values():
        assert token_id >= BYTE_VOCAB_SIZE
        assert token_id not in merge_ids


def test_decode_rejects_an_unknown_id(trained):
    with pytest.raises(KeyError):
        trained.decode([10**6])


# --- saving and loading ------------------------------------------------------


def test_save_and_load_preserves_encoding(trained, tmp_path):
    path = tmp_path / "vocab.json"
    trained.save(path)
    reloaded = BPETokenizer.load(path)

    assert reloaded.merges == trained.merges
    assert reloaded.vocab_size == trained.vocab_size
    assert reloaded.encode(SAMPLE_CORPUS) == trained.encode(SAMPLE_CORPUS)


def test_saved_file_is_readable_json(trained, tmp_path):
    path = tmp_path / "vocab.json"
    trained.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["special_tokens"] == list(SPECIAL_TOKENS)
    assert len(payload["merges"]) == len(trained.merges)


# --- corpus loading and measurement ------------------------------------------


def test_extract_strings_collects_only_strings():
    value = {
        "name": "Leia",
        "lines": ["Help me", {"target": "Obi-Wan"}, 42, None],
        "active": True,
    }
    assert extract_strings(value) == ["Leia", "Help me", "Obi-Wan"]


@pytest.mark.parametrize("value", [None, 7, 3.14, True])
def test_extract_strings_ignores_non_strings(value):
    assert extract_strings(value) == []


def test_load_corpus_reads_the_dataset(dataset_dir):
    corpus = load_corpus()
    assert "the Force" in corpus
    assert "A New Hope" in corpus


def test_measure_reports_the_four_plan_metrics(trained):
    stats = measure(trained, SAMPLE_CORPUS)

    assert stats["vocab_size"] == trained.vocab_size
    assert stats["tokens"] > 0
    assert stats["tokens_per_word"] > 0
    assert stats["unk_rate"] == 0.0
    assert stats["round_trips"] is True


def test_measure_handles_empty_text(trained):
    stats = measure(trained, "")
    assert stats["tokens"] == 0
    assert stats["tokens_per_word"] == 0.0
    assert stats["unk_rate"] == 0.0


def test_main_trains_and_reports(dataset_dir, capsys):
    main()
    output = capsys.readouterr().out

    assert "training text" in output
    assert "held-out text" in output
    assert "round trips       True" in output
    assert (dataset_dir / "bpe_vocab.json").exists()
