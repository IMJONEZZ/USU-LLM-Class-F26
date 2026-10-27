"""
Tests for the loss/perplexity evaluator in src/evaluator.py, updated for
this assignment's causal-LM evaluation (measuring loss on held-out Star
Wars dialogue), replacing the earlier BERT/SST-2 classification evaluator.
"""

import json
import math

import pytest
import torch

from src.dataloader import TEST_MOVIE
from src.evaluator import compute_loss, load_test_text

# ---------------------------------------------------------------------------
# FAKE MODEL / TOKENIZER
# Minimal stand-ins that mimic just enough of a Hugging Face causal LM's
# interface for compute_loss() to work correctly against them, without
# downloading a real model or needing a GPU.
# ---------------------------------------------------------------------------


class FakeOutputs:
    """Mimics a Hugging Face causal LM output object with a `.loss` attribute."""

    def __init__(self, loss):
        self.loss = loss


class FakeModel:
    """A fake causal LM that always reports a fixed loss value."""

    def __init__(self, fixed_loss=1.0):
        self.fixed_loss = fixed_loss

    def eval(self):
        pass

    def __call__(self, input_ids, labels=None):
        return FakeOutputs(torch.tensor(self.fixed_loss))


class FakeEncodings:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class FakeTokenizer:
    """A fake tokenizer that returns one token ID per character."""

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % 100 for c in text]
        return FakeEncodings(torch.tensor([ids]))


# ---------------------------------------------------------------------------
# compute_loss() TESTS
# ---------------------------------------------------------------------------


def test_compute_loss_returns_loss_and_perplexity():
    """compute_loss should return both a loss value and its exponential (perplexity)."""
    model = FakeModel(fixed_loss=2.0)
    tokenizer = FakeTokenizer()
    text = "a" * 300

    avg_loss, perplexity = compute_loss(
        model, tokenizer, text, max_length=50, stride=25, device="cpu"
    )

    assert avg_loss == 2.0
    assert math.isclose(perplexity, math.exp(2.0), rel_tol=1e-6)


def test_compute_loss_too_short_text_raises_error():
    """Text too short to form even one chunk should raise a clear error, not silently return."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    text = "short"

    with pytest.raises(ValueError):
        compute_loss(model, tokenizer, text, max_length=50, stride=25, device="cpu")


def test_compute_loss_perplexity_is_exp_of_loss():
    """Perplexity should always equal e^(loss), the standard relationship between the two."""
    model = FakeModel(fixed_loss=1.5)
    tokenizer = FakeTokenizer()
    text = "a" * 300

    avg_loss, perplexity = compute_loss(
        model, tokenizer, text, max_length=50, stride=25, device="cpu"
    )

    assert math.isclose(perplexity, math.exp(avg_loss), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# load_test_text() TEST
# Mocked via a temp file, so this doesn't depend on the real dataset existing.
# ---------------------------------------------------------------------------


def test_load_test_text_only_includes_test_movie(tmp_path):
    """load_test_text should only include scenes from the designated test movie."""
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

    result = load_test_text(str(fake_file))

    assert "I am a Jedi." in result
    assert "Hello there." not in result
