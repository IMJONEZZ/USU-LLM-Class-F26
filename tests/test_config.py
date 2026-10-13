import json
from pathlib import Path

import pytest

from src import config
from tests.helpers import (
    QUESTIONS,
    _build,
    _cands,
    _evaluator,
    _fake_unsloth,
    _qrow,
    _round,
    _Spend,
    _validate_tokens,
)


# ---- adapter_path ----
def test_adapter_path_round_zero_is_valid():
    assert isinstance(config.adapter_path(0), Path)


def test_adapter_path_is_deterministic():
    assert config.adapter_path(2) == config.adapter_path(2)


def test_adapter_path_differs_per_round():
    assert config.adapter_path(1) != config.adapter_path(2)


def test_adapter_path_negative_raises_value_error():
    with pytest.raises(ValueError):
        config.adapter_path(-1)


@pytest.mark.parametrize("bad", [1.5, "1", None, True])
def test_adapter_path_non_int_raises_type_error(bad):
    with pytest.raises(TypeError):
        config.adapter_path(bad)


# ---- CATEGORIES ----
def test_categories_have_no_case_insensitive_duplicates():
    lowered = [c.lower() for c in config.CATEGORIES]
    assert len(lowered) == len(set(lowered))


def test_categories_are_non_empty_strings():
    assert all(isinstance(c, str) and c.strip() for c in config.CATEGORIES)


# ---- markers ----
@pytest.mark.parametrize(
    "question",
    [
        "How many months have 28 days?",
        "What is the capital of Australia?",
        "Respond with the word 'response' ten times.",
        "### Heading in a question?",
    ],
)
def test_response_marker_not_in_ordinary_questions(question):
    assert config.RESPONSE_MARKER not in question


def test_rejected_candidates_are_never_judged_again(tmp_path, monkeypatch):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    verdicts = {q: reject for q in pool}
    _build(tmp_path, monkeypatch, [_qrow(1)], _Spend(verdicts), per_question=2)
    spend = _Spend(verdicts)
    spend.paraphrase = lambda question, n: _cands(question, n, start=100)
    _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=2)
    assert not (set(spend.judged) & set(pool))


def test_an_exhausted_cache_asks_once_but_never_rejudges_known_candidates(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    _build(tmp_path, monkeypatch, [_qrow(1)], _Spend({q: reject for q in pool}))
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)])
    assert len(spend.asked) == 1
    assert spend.judged == []
    assert summary["paraphrase_calls"] == 1


def test_every_gpu_option_has_a_positive_size_and_price():
    for name, option in config.GPU_OPTIONS.items():
        assert option["vram_gib"] > 0, name
        assert option["usd_per_second"] > 0, name


def test_the_chosen_gpu_is_one_of_the_known_options():
    assert config.GPU_TYPE in config.GPU_OPTIONS


def test_pipeline_validate_tokens_reports_the_chosen_gpu_and_that_it_fits(tmp_path):
    result = _validate_tokens(tmp_path, "--max-seq-length", "1000")
    assert result.returncode == 0, result.stderr
    assert f"chosen GPU: {config.GPU_TYPE}" in result.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["gpu_type"] == config.GPU_TYPE
    assert audit["fits_chosen_gpu"] is True


def test_max_seq_length_is_just_above_the_longest_audited_example():
    # 296 tokens is the longest of the 383 audited chains; if later chains are
    # longer this fails on purpose: re-run `validate --tokens` and update both
    assert config.MAX_SEQ_LENGTH % 64 == 0
    assert 296 <= config.MAX_SEQ_LENGTH < 296 + 64


# ---- M8: planning helpers and config ----
def test_adapter_path_accepts_a_root_for_the_modal_volume(tmp_path):
    assert config.adapter_path(2, root=tmp_path) == tmp_path / "round_2"
    assert config.adapter_path(2) == config.ADAPTER_DIR / "round_2"


def test_epoch_evaluator_keeps_the_same_category_keys_every_epoch(monkeypatch):
    _fake_unsloth(monkeypatch)
    evaluator = _evaluator()
    first = evaluator.evaluate_epoch("m", 1, 1.0, 1.0)
    second = evaluator.evaluate_epoch("m", 2, 0.9, 0.9)
    assert (
        list(first["per_category"]) == list(second["per_category"]) == config.CATEGORIES
    )
    assert first["per_category"]["time_sensitive"] == "undefined"
    assert [e["epoch"] for e in evaluator.epochs] == [1, 2]


def test_the_repeat_cap_defaults_to_the_configured_value():
    result = _round(extra_rows=None, replay_fraction=0)
    assert result["new"] == 5 * config.ROUND_REPEAT_CAP


def test_config_has_the_round_settings():
    assert config.ROUND_REPEAT_CAP >= 1
    assert config.MAX_ROUNDS == 3
