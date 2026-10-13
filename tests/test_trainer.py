"""
Tests for src/trainer.py.

Training itself requires a GPU and the Unsloth stack (CI has no GPU), so
these tests focus on the parts of trainer.py that don't require a GPU --
loading training examples and building the chat-formatted dataset -- and
mock out anything that would actually load a model or run training.
"""

import json

from src.trainer import build_training_dataset, load_training_examples

# ---------------------------------------------------------------------------
# load_training_examples() TESTS
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_load_training_examples_reads_all_examples(tmp_path):
    """load_training_examples should read every example from the jsonl file."""
    fake_examples = [
        {"category": "scaling_rate_trap", "question": "Q1", "answer": "A1"},
        {"category": "letter_counting", "question": "Q2", "answer": "A2"},
    ]
    fake_file = tmp_path / "fake_training.jsonl"
    fake_file.write_text(
        "\n".join(json.dumps(ex) for ex in fake_examples), encoding="utf-8"
    )

    result = load_training_examples(str(fake_file))

    assert len(result) == 2
    assert result[0]["question"] == "Q1"
    assert result[1]["category"] == "letter_counting"


def test_load_training_examples_empty_file_returns_empty_list(tmp_path):
    """An empty training file should produce an empty list, not an error."""
    fake_file = tmp_path / "empty.jsonl"
    fake_file.write_text("", encoding="utf-8")

    result = load_training_examples(str(fake_file))

    assert result == []


def test_load_training_examples_preserves_all_fields(tmp_path):
    """Each loaded example should retain its question, answer, and category fields."""
    fake_examples = [
        {
            "category": "wordplay_ambiguity",
            "question": "Is X heavier?",
            "answer": "No, because Y.",
        },
    ]
    fake_file = tmp_path / "fake_training.jsonl"
    fake_file.write_text(json.dumps(fake_examples[0]), encoding="utf-8")

    result = load_training_examples(str(fake_file))

    assert result[0]["question"] == "Is X heavier?"
    assert result[0]["answer"] == "No, because Y."
    assert result[0]["category"] == "wordplay_ambiguity"


# ---------------------------------------------------------------------------
# build_training_dataset() TESTS
# This function needs a tokenizer, which we mock rather than downloading a
# real model, since CI has no GPU and shouldn't need network access.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """
    A minimal fake tokenizer that mimics apply_chat_template's interface,
    just enough to exercise build_training_dataset's logic without needing
    a real tokenizer or GPU.
    """

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=False,
        max_length=None,
        truncation=False,
    ):
        # Simple fake tokenization: one "token" per word across all messages
        text = " ".join(m["content"] for m in messages)
        token_ids = [hash(word) % 1000 for word in text.split()]
        if max_length is not None and truncation:
            token_ids = token_ids[:max_length]
        return token_ids


def test_build_training_dataset_produces_one_example_per_input():
    """Each input example should produce exactly one formatted training example."""
    tokenizer = FakeTokenizer()
    examples = [
        {"question": "What is 2 plus 2?", "answer": "It is 4."},
        {"question": "How many days in a week?", "answer": "There are 7."},
    ]

    dataset = build_training_dataset(tokenizer, examples, max_seq_length=50)

    assert len(dataset) == 2


def test_build_training_dataset_respects_max_seq_length():
    """No example's input_ids should exceed max_seq_length after truncation."""
    tokenizer = FakeTokenizer()
    examples = [
        {"question": "word " * 100, "answer": "word " * 100},
    ]

    dataset = build_training_dataset(tokenizer, examples, max_seq_length=20)

    assert len(dataset[0]["input_ids"]) <= 20


def test_build_training_dataset_empty_examples_list():
    """An empty list of examples should produce an empty dataset, not an error."""
    tokenizer = FakeTokenizer()

    dataset = build_training_dataset(tokenizer, [], max_seq_length=50)

    assert len(dataset) == 0


# ---------------------------------------------------------------------------
# train_model() / load_model_for_training() -- GPU/Unsloth-dependent
# Intentionally NOT tested here, since they require a GPU and the Unsloth
# stack, which this CI environment does not have.
# ---------------------------------------------------------------------------


def test_trainer_module_is_importable_without_a_gpu():
    """
    Confirm that importing src.trainer doesn't itself require a GPU or
    trigger model loading -- only calling train_model()/
    load_model_for_training() should need one.
    """
    import src.trainer  # noqa: F401
