"""
Tests for src/trainer.py.

Training itself requires a GPU and the Unsloth/peft/transformers stack
installed inside a specific Docker environment (per Assignment 4's note
that CI/CD won't have a GPU available). These tests focus on the parts of
trainer.py that don't require a GPU -- loading and chunking training data
-- and mock out anything that would actually load a model or run training,
so this file can run cleanly in CI without a GPU.
"""

import json

import torch

from src.dataloader import TEST_MOVIE, VAL_MOVIE
from src.trainer import build_training_dataset, load_training_text

# ---------------------------------------------------------------------------
# load_training_text() TESTS
# Mocked via a temp file, so this doesn't depend on the real dataset existing,
# and reuses the same movie-based split logic tested in test_dataloader.py.
# ---------------------------------------------------------------------------


def test_load_training_text_excludes_validation_movie(tmp_path):
    """Scenes from the validation movie should not appear in training text."""
    fake_scenes = [
        {
            "movie": "A New Hope 4K77",
            "turns": [
                {"speaker": "Luke", "text": "I want to learn the ways of the Force."}
            ],
        },
        {
            "movie": VAL_MOVIE,
            "turns": [{"speaker": "Anakin", "text": "I will do what I must."}],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(
        "\n".join(json.dumps(s) for s in fake_scenes), encoding="utf-8"
    )

    result = load_training_text(str(fake_file))

    assert "I want to learn the ways of the Force." in result
    assert "I will do what I must." not in result


def test_load_training_text_excludes_test_movie(tmp_path):
    """Scenes from the test movie should not appear in training text."""
    fake_scenes = [
        {
            "movie": "A New Hope 4K77",
            "turns": [{"speaker": "Luke", "text": "Hello there."}],
        },
        {"movie": TEST_MOVIE, "turns": [{"speaker": "Luke", "text": "I am a Jedi."}]},
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(
        "\n".join(json.dumps(s) for s in fake_scenes), encoding="utf-8"
    )

    result = load_training_text(str(fake_file))

    assert "Hello there." in result
    assert "I am a Jedi." not in result


def test_load_training_text_includes_multiple_non_held_out_titles(tmp_path):
    """Scenes from several different non-held-out titles should all appear in training text."""
    fake_scenes = [
        {
            "movie": "A New Hope 4K77",
            "turns": [{"speaker": "Leia", "text": "Help me, Obi-Wan Kenobi."}],
        },
        {
            "movie": "The Empire Strikes Back 4K80",
            "turns": [{"speaker": "Yoda", "text": "Do or do not."}],
        },
        {
            "movie": "Rogue One",
            "turns": [{"speaker": "Jyn", "text": "Rebellions are built on hope."}],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(
        "\n".join(json.dumps(s) for s in fake_scenes), encoding="utf-8"
    )

    result = load_training_text(str(fake_file))

    assert "Help me, Obi-Wan Kenobi." in result
    assert "Do or do not." in result
    assert "Rebellions are built on hope." in result


def test_load_training_text_empty_file_returns_empty_string(tmp_path):
    """An empty dataset file should produce an empty training text string."""
    fake_file = tmp_path / "empty.jsonl"
    fake_file.write_text("", encoding="utf-8")

    result = load_training_text(str(fake_file))

    assert result == ""


# ---------------------------------------------------------------------------
# build_training_dataset() TESTS
# This function needs a tokenizer, which we mock rather than downloading a
# real model, since CI has no GPU and shouldn't need network access to
# Hugging Face just to run unit tests.
# ---------------------------------------------------------------------------


class FakeTokenizerOutput:
    """Mimics the object returned by calling a Hugging Face tokenizer."""

    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    """
    A minimal fake tokenizer: encodes text as one token ID per character,
    just enough to exercise the chunking logic in build_training_dataset
    without needing a real tokenizer or GPU.
    """

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) for c in text]
        return FakeTokenizerOutput(torch.tensor([ids]))


def test_build_training_dataset_produces_fixed_length_chunks():
    """Each example in the dataset should have exactly max_seq_length token IDs."""
    tokenizer = FakeTokenizer()
    text = "a" * 500  # long enough to produce several chunks
    dataset = build_training_dataset(tokenizer, text, max_seq_length=50)

    assert len(dataset) > 0
    for example in dataset:
        assert len(example["input_ids"]) == 50


def test_build_training_dataset_too_short_text_produces_no_chunks():
    """Text shorter than max_seq_length should produce zero chunks, not an error."""
    tokenizer = FakeTokenizer()
    text = "short"
    dataset = build_training_dataset(tokenizer, text, max_seq_length=50)

    assert len(dataset) == 0


def test_build_training_dataset_chunks_overlap_with_stride():
    """With max_seq_length=50, stride is max_seq_length // 4 = 12; chunks should overlap."""
    tokenizer = FakeTokenizer()
    text = "ab" * 100
    dataset = build_training_dataset(tokenizer, text, max_seq_length=50)

    assert len(dataset) > 1
    first_chunk = dataset[0]["input_ids"]
    second_chunk = dataset[1]["input_ids"]
    assert first_chunk[12:] == second_chunk[:38]


# ---------------------------------------------------------------------------
# train_model() / load_model_for_training() -- GPU/Unsloth-dependent
# These are intentionally NOT tested here, since they require a GPU and the
# Unsloth stack, which this CI environment does not have. This mirrors the
# assignment's own note: "you will also have to account for this in your
# tests as your CI/CD will likely fail since it won't have a GPU."
# ---------------------------------------------------------------------------


def test_trainer_module_is_importable_without_a_gpu():
    """
    Confirm that importing src.trainer doesn't itself require a GPU or trigger
    model loading -- only calling train_model()/load_model_for_training()
    should need one. This guards against accidentally moving GPU-dependent
    code to module level, which would break CI entirely.
    """
    import src.trainer  # noqa: F401
