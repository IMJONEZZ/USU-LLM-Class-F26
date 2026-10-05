import numpy as np
import pandas as pd
import pytest

from src.evaluator import evaluate_model, load_eval_subset
from src.prompts import format_prompt

CATEGORIES = ["HR", "BPO"]


class FakeEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[text] for text in texts], dtype=float)


def make_generate_fn(outputs):
    batches = []
    remaining_outputs = iter(outputs)

    def generate_fn(prompts):
        batches.append(list(prompts))
        return [next(remaining_outputs) for _ in prompts]

    generate_fn.batches = batches
    return generate_fn


def make_eval_resumes():
    return pd.DataFrame(
        {"Resume_str": ["r1", "r2", "r3"], "Category": ["HR", "HR", "BPO"]}
    )


def make_embedding_model():
    return FakeEmbeddingModel({"HR": [1, 0], "BPO": [0, 1], "banana": [1, 1]})


def test_evaluate_model_hand_checked_metrics():
    generate_fn = make_generate_fn(["HR", "BPO", "banana"])

    result = evaluate_model(
        generate_fn,
        make_eval_resumes(),
        CATEGORIES,
        make_embedding_model(),
        batch_size=2,
    )

    assert result["n"] == 3
    assert result["mean_semantic_similarity"] == pytest.approx((1 + 0 + 2**-0.5) / 3)
    assert result["valid_category_rate"] == pytest.approx(2 / 3)
    assert result["exact_match_rate"] == pytest.approx(1 / 3)


def test_evaluate_model_sends_format_prompt_text_in_batches_of_batch_size():
    generate_fn = make_generate_fn(["HR", "BPO", "banana"])

    evaluate_model(
        generate_fn,
        make_eval_resumes(),
        CATEGORIES,
        make_embedding_model(),
        batch_size=2,
    )

    assert generate_fn.batches == [
        [format_prompt("r1", CATEGORIES), format_prompt("r2", CATEGORIES)],
        [format_prompt("r3", CATEGORIES)],
    ]


def test_load_eval_subset_returns_n_rows():
    subset = load_eval_subset(make_eval_resumes(), n=2, seed=0)
    assert len(subset) == 2


def test_load_eval_subset_is_reproducible_with_same_seed():
    first = load_eval_subset(make_eval_resumes(), n=2, seed=7)
    second = load_eval_subset(make_eval_resumes(), n=2, seed=7)
    pd.testing.assert_frame_equal(first, second)
