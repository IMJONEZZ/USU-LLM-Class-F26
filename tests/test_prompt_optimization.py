"""
Tests for src/prompt_optimization.py.

Tests focus on the parts that don't require API calls: dataset loading,
splitting, and the exact-match metric. The actual DSPy optimization
(BootstrapFewShot/MIPROv2) requires a real LM and API key, so it's not
tested directly here -- those functions are marked # pragma: no cover,
consistent with how GPU/API-dependent code has been handled all semester.
"""

import json

import dspy

from src.prompt_optimization import (
    build_dspy_examples,
    exact_match_metric,
    extract_labeled_lines,
    split_dataset,
)

# ---------------------------------------------------------------------------
# extract_labeled_lines() TESTS
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_extract_labeled_lines_filters_short_lines(tmp_path):
    """Lines shorter than min_length should be excluded."""
    fake_scenes = [
        {
            "movie": "A New Hope",
            "turns": [
                {"speaker": "Luke", "text": "Yes."},
                {
                    "speaker": "Leia",
                    "text": "Help me, Obi-Wan Kenobi, you're my only hope.",
                },
            ],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(json.dumps(fake_scenes[0]), encoding="utf-8")

    lines = extract_labeled_lines(str(fake_file), min_length=15)

    assert len(lines) == 1
    assert lines[0]["speaker"] == "Leia"


def test_extract_labeled_lines_respects_max_lines(tmp_path):
    """extract_labeled_lines should stop once max_lines is reached."""
    fake_scenes = [
        {
            "movie": "A New Hope",
            "turns": [
                {"speaker": "Luke", "text": f"This is a long enough line {i}"}
                for i in range(10)
            ],
        },
    ]
    fake_file = tmp_path / "fake_scenes.jsonl"
    fake_file.write_text(json.dumps(fake_scenes[0]), encoding="utf-8")

    lines = extract_labeled_lines(str(fake_file), max_lines=3, min_length=5)

    assert len(lines) == 3


def test_extract_labeled_lines_empty_file_returns_empty_list(tmp_path):
    """An empty dataset file should produce an empty list, not an error."""
    fake_file = tmp_path / "empty.jsonl"
    fake_file.write_text("", encoding="utf-8")

    lines = extract_labeled_lines(str(fake_file))

    assert lines == []


# ---------------------------------------------------------------------------
# split_dataset() TESTS
# ---------------------------------------------------------------------------


def test_split_dataset_produces_correct_sizes():
    """Train/dev/test splits should roughly match the requested fractions."""
    lines = [{"speaker": f"Speaker{i}", "text": f"Line {i}"} for i in range(100)]

    train, dev, test = split_dataset(lines, train_frac=0.6, dev_frac=0.2)

    assert len(train) == 60
    assert len(dev) == 20
    assert len(test) == 20


def test_split_dataset_no_overlap():
    """No line should appear in more than one split."""
    lines = [{"speaker": f"Speaker{i}", "text": f"Line {i}"} for i in range(50)]

    train, dev, test = split_dataset(lines)

    train_texts = {line["text"] for line in train}
    dev_texts = {line["text"] for line in dev}
    test_texts = {line["text"] for line in test}

    assert train_texts.isdisjoint(dev_texts)
    assert train_texts.isdisjoint(test_texts)
    assert dev_texts.isdisjoint(test_texts)


def test_split_dataset_is_reproducible_with_same_seed():
    """Using the same seed should produce the same split every time."""
    lines = [{"speaker": f"Speaker{i}", "text": f"Line {i}"} for i in range(50)]

    train1, dev1, test1 = split_dataset(lines, seed=17)
    train2, dev2, test2 = split_dataset(lines, seed=17)

    assert train1 == train2
    assert dev1 == dev2
    assert test1 == test2


def test_split_dataset_includes_every_line_exactly_once():
    """Every original line should appear in exactly one of the three splits."""
    lines = [{"speaker": f"Speaker{i}", "text": f"Line {i}"} for i in range(30)]

    train, dev, test = split_dataset(lines)

    all_split_texts = [line["text"] for line in train + dev + test]
    original_texts = [line["text"] for line in lines]

    assert sorted(all_split_texts) == sorted(original_texts)


# ---------------------------------------------------------------------------
# exact_match_metric() TESTS
# ---------------------------------------------------------------------------


class FakePrediction:
    """A minimal stand-in for a dspy.Prediction object with a .speaker attribute."""

    def __init__(self, speaker):
        self.speaker = speaker


def test_exact_match_metric_scores_correct_match():
    """An exact match (same case) should score 1.0."""
    example = dspy.Example(speaker="Obi-Wan Kenobi")
    prediction = FakePrediction("Obi-Wan Kenobi")

    assert exact_match_metric(example, prediction) == 1.0


def test_exact_match_metric_scores_mismatch():
    """A wrong guess should score 0.0."""
    example = dspy.Example(speaker="Obi-Wan Kenobi")
    prediction = FakePrediction("Captain Kirk")

    assert exact_match_metric(example, prediction) == 0.0


def test_exact_match_metric_is_case_insensitive():
    """Matching should ignore letter case differences."""
    example = dspy.Example(speaker="Obi-Wan Kenobi")
    prediction = FakePrediction("obi-wan kenobi")

    assert exact_match_metric(example, prediction) == 1.0


def test_exact_match_metric_ignores_whitespace():
    """Matching should ignore leading/trailing whitespace."""
    example = dspy.Example(speaker="Obi-Wan Kenobi")
    prediction = FakePrediction("  Obi-Wan Kenobi  ")

    assert exact_match_metric(example, prediction) == 1.0


def test_exact_match_metric_normalizes_accented_characters():
    """Matching should treat accented and unaccented versions of a name as equal."""
    example = dspy.Example(speaker="Padme Amidala")
    prediction = FakePrediction("Padmé Amidala")

    assert exact_match_metric(example, prediction) == 1.0


# ---------------------------------------------------------------------------
# build_dspy_examples() TESTS
# ---------------------------------------------------------------------------


def test_build_dspy_examples_converts_correctly():
    """Each line dict should become a dspy.Example with the correct fields."""
    lines = [{"speaker": "Yoda", "text": "Do or do not."}]

    examples = build_dspy_examples(lines)

    assert len(examples) == 1
    assert examples[0].speaker == "Yoda"
    assert examples[0].dialogue_line == "Do or do not."


def test_build_dspy_examples_marks_dialogue_line_as_input():
    """The dialogue_line field should be marked as the input key."""
    lines = [{"speaker": "Yoda", "text": "Do or do not."}]

    examples = build_dspy_examples(lines)

    assert list(examples[0].inputs().keys()) == ["dialogue_line"]


def test_build_dspy_examples_empty_list_returns_empty_list():
    """An empty list of lines should produce an empty list of examples."""
    examples = build_dspy_examples([])

    assert examples == []


# ---------------------------------------------------------------------------
# Module importability check
# ---------------------------------------------------------------------------


def test_prompt_optimization_module_is_importable_without_a_gpu_or_api_key():
    """
    Confirm that importing src.prompt_optimization doesn't itself require
    an API key or trigger any model calls -- only running the module's
    __main__ block should need that.
    """
    import src.prompt_optimization  # noqa: F401
