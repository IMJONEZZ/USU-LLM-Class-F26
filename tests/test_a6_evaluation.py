"""Test development coverage, answer errors and provenance without a GPU."""

import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from src.a6_evaluation import (
    assert_split_integrity,
    dataset_digest,
    development_metrics,
    score_development_answer,
    split_manifest,
    stratified_family_split,
    validate_completion_batch,
)
from src.trainer import TrainingConfig, load_own_examples, training_arguments

ROOT = Path(__file__).parents[1]
DATA = ROOT / "assignment_6/targeted_examples.jsonl"


def examples():
    return load_own_examples(DATA)


def synthetic(family, skill="count", tag="Q9", variant=0):
    return {
        "id": f"{family}-{variant}",
        "family": family,
        "tag": tag,
        "skill": skill,
        "messages": [
            {"role": "user", "content": f"Example {family} variant {variant}?"},
            {"role": "assistant", "content": "1"},
        ],
        "evaluation": {"kind": "number", "value": "1"},
    }


def test_real_split_represents_every_skill_and_keeps_families_together():
    rows = examples()
    train, dev = stratified_family_split(rows)
    buckets = {(r["tag"], r["skill"]) for r in rows}
    assert {(r["tag"], r["skill"]) for r in dev} == buckets
    assert {(r["tag"], r["skill"]) for r in train} == buckets
    assert_split_integrity(train, dev)
    assert (train, dev) == stratified_family_split(rows)
    assert len(train) + len(dev) == len(rows)
    assert any(r["skill"] == "currency_vs_mass" for r in dev)
    assert any(r["skill"] == "quantity_bound" for r in dev)


def test_repeated_word_or_scenario_variants_cannot_cross_split():
    rows = [synthetic(str(i), variant=v) for i in range(10) for v in range(2)]
    train, dev = stratified_family_split(rows)
    assert {r["family"] for r in train}.isdisjoint(r["family"] for r in dev)


@pytest.mark.parametrize("fraction", [0, 1, -0.1])
def test_invalid_split_fraction_rejected(fraction):
    with pytest.raises(ValueError):
        stratified_family_split(
            [synthetic("a"), synthetic("b")], development_fraction=fraction
        )


def test_one_family_cannot_support_independent_development():
    with pytest.raises(ValueError, match="independent families"):
        stratified_family_split([synthetic("a"), synthetic("a", variant=1)])


def test_split_integrity_rejects_prompt_and_family_leakage():
    with pytest.raises(ValueError):
        assert_split_integrity([], [synthetic("a")])
    with pytest.raises(ValueError, match="prompt"):
        assert_split_integrity([synthetic("a")], [synthetic("a")])
    with pytest.raises(ValueError, match="family"):
        assert_split_integrity([synthetic("a")], [synthetic("a", variant=1)])


def test_all_reference_answers_satisfy_their_declared_diagnostic_rules():
    for row in examples():
        assert score_development_answer(row, row["messages"][-1]["content"]), row


def test_numeric_rule_rejects_wrong_verdict_even_if_correct_number_appears_later():
    row = synthetic("a")
    row["evaluation"] = {"kind": "number", "value": "7", "unit": "minutes"}
    assert score_development_answer(row, "7 minutes.")
    assert not score_development_answer(row, "2 minutes, though the prompt said 7.")
    assert not score_development_answer(row, "7 days.")
    assert not score_development_answer(row, "I do not know.")


def test_letter_rule_rejects_correct_count_with_wrong_spelling():
    row = synthetic("a")
    row["evaluation"] = {"kind": "letters", "word": "yellow", "letter": "l", "count": 2}
    assert score_development_answer(
        row, 'y-e-l-l-o-w. There are 2 L characters in "yellow".'
    )
    assert score_development_answer(row, "2")
    assert not score_development_answer(
        row, 'y-e-l-l-o-r. There are 2 L characters in "yellow".'
    )
    assert not score_development_answer(
        row, 'y-e-l-l-o-w. There are 3 L characters in "yellow".'
    )
    row["evaluation"] = {"kind": "copy", "word": "yellow"}
    assert score_development_answer(row, "y-e-l-l-o-w.")
    assert not score_development_answer(row, "y-e-l-l-o-r.")


def test_safety_rule_rejects_contradictory_yes_and_no():
    row = synthetic("a")
    row["evaluation"] = {"kind": "safety", "verdict": "yes", "required": [["water"]]}
    assert score_development_answer(row, "It is confined to water. So yes, it is safe.")
    assert not score_development_answer(
        row, "No. It stays in water. So yes, it is safe."
    )
    assert not score_development_answer(row, "Yes, it is safe.")


def test_terms_rules_reject_missing_requirements_and_explicit_denial():
    row = synthetic("a")
    row["evaluation"] = {"kind": "terms", "required": [["birthday", "birth day"]]}
    assert score_development_answer(row, "It is a birthday party.")
    assert not score_development_answer(row, "It is not a birthday.")
    assert not score_development_answer(row, "It is a wedding.")
    row["evaluation"]["forbidden"] = ["wedding"]
    assert not score_development_answer(row, "Birthday or wedding.")
    assert not score_development_answer(row, "")
    row["evaluation"] = {"kind": "unknown"}
    with pytest.raises(ValueError):
        score_development_answer(row, "A response.")


def test_macro_score_weights_question_groups_equally():
    rows = [synthetic("a", tag="Q1")] + [synthetic(str(i), tag="Q9") for i in range(9)]
    metrics, details = development_metrics(rows, ["1"] + ["2"] * 9)
    assert metrics["skill_match_accuracy"] == 0.5
    assert len(details) == 10
    with pytest.raises(ValueError):
        development_metrics(rows, [])


def test_manifest_preserves_rows_and_hash_changes_with_answer():
    train, dev = stratified_family_split(examples())
    manifest = split_manifest(train, dev)
    assert manifest["training"] == train
    assert manifest["development"] == dev
    altered = [synthetic("a")]
    original_hash = dataset_digest(altered)
    altered[0]["messages"][-1]["content"] = "2"
    assert dataset_digest(altered) != original_hash


class Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


def test_actual_collator_contract_checks_masking_and_labels():
    batch = {
        "input_ids": Array([[10, 11, 12, 0]]),
        "labels": Array([[-100, -100, 12, -100]]),
        "attention_mask": Array([[1, 1, 1, 0]]),
    }
    assert validate_completion_batch(batch) == {
        "supervised_tokens": 1,
        "masked_tokens": 3,
    }
    for labels in [
        [10, 11, 12, -100],
        [-100] * 4,
        [-100, -100, 12, 0],
        [-100, -100, 13, -100],
    ]:
        batch["labels"] = Array([labels])
        with pytest.raises(ValueError):
            validate_completion_batch(batch)


def test_checkpoint_controls_validate_and_select_generation_metric():
    config = TrainingConfig(
        max_length=1024,
        output_dir="test",
        epochs=3,
        checkpoint_metric="eval_skill_match_accuracy",
        warmup_ratio=0.05,
        lora_dropout=0.05,
    )
    args = training_arguments(config, 428)
    assert args["greater_is_better"]
    assert args["metric_for_best_model"] == "eval_skill_match_accuracy"
    assert args["save_total_limit"] == 3
    for kwargs in [
        {"lora_dropout": 1},
        {"weight_decay": -1},
        {"max_grad_norm": 0},
        {"checkpoint_metric": "invalid"},
        {"lr_scheduler_type": "invalid"},
    ]:
        with pytest.raises(ValueError):
            TrainingConfig(max_length=1024, output_dir="test", **kwargs)


def test_regenerated_dataset_matches_and_q9_no_longer_dominates():
    path = ROOT / "assignment_6/make_targeted_dataset.py"
    spec = importlib.util.spec_from_file_location("a6_generator", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    saved = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert saved == generator.all_examples()
    counts = Counter(row["tag"] for row in saved)
    assert counts["Q9"] / len(saved) < 0.2
    assert counts["Q6"] == 43
    assert not any(
        "Do I need to avoid" in r["messages"][0]["content"]
        for r in saved
        if r["tag"] == "Q5"
    )


def test_occasion_matching_accepts_a_name_without_copying_reference_explanation():
    row = next(r for r in examples() if "Bastille" in r["messages"][-1]["content"])
    assert score_development_answer(row, "It is Bastille Day.")
    assert not score_development_answer(row, "It is a wedding.")
    row = next(
        r
        for r in examples()
        if r["tag"] == "Q3" and "baby shower" in r["messages"][-1]["content"]
    )
    assert score_development_answer(row, "It is a baby shower.")


def test_pennyweight_comparison_rejects_reversed_conclusion():
    # Preserve this historical matcher regression even though v6 prunes the row.
    row = synthetic("pennyweight", skill="unit_distinction", tag="Q2")
    row["messages"][0]["content"] = "Which weighs more, a US penny or a pennyweight?"
    row["evaluation"] = {"kind": "terms", "required": [["A US penny"]]}
    assert not score_development_answer(
        row, "A US penny is worth a cent. A pennyweight is heavier."
    )
    assert score_development_answer(row, "A US penny is heavier than a pennyweight.")


def test_supply_conversion_accepts_consistent_leading_expression_but_not_wrong_verdict():
    row = next(
        r
        for r in examples()
        if "100 grams of tea leaves" in r["messages"][0]["content"]
    )
    assert score_development_answer(
        row,
        "No more than 100 grams / 2 grams per cup = 50 cups. The tea would run out.",
    )
    assert not score_development_answer(
        row,
        "At most 30 cups. 100 grams / 2 grams per cup = 50 cups; the supply runs out.",
    )
    assert not score_development_answer(
        row,
        "No more than 100 grams / 3 grams per cup = 50 cups. The tea would run out.",
    )


def test_currency_answers_reject_contradictory_equal_weight_fallback():
    rows = [r for r in examples() if r["skill"] == "currency_vs_mass"]
    assert len(rows) == 22
    for row in rows:
        reference = row["messages"][-1]["content"]
        assert "If you mean the physical coin or banknote" in reference
        assert score_development_answer(row, reference)
        assert not score_development_answer(
            row, reference + " They weigh the same because both are units."
        )


def test_q2_cleanup_retains_explicit_mass_controls_without_conversion_noise():
    rows = [r for r in examples() if r["tag"] == "Q2"]
    assert Counter(r["skill"] for r in rows) == {
        "currency_vs_mass": 22,
        "equal_mass": 4,
        "unit_distinction": 6,
    }
    for row in rows:
        prompt = row["messages"][0]["content"].lower()
        assert not any(
            term in prompt
            for term in (
                "pennyweight",
                "fluid ounce",
                "stone",
                "karat",
                "pint",
                "liter",
            )
        )
        if row["skill"] == "equal_mass":
            assert "not currency amounts" in row["messages"][-1]["content"]


def test_counting_rule_requires_correct_word_without_extra_spelling():
    row = synthetic("yellow")
    row["evaluation"] = {
        "kind": "letters",
        "word": "yellow",
        "letter": "l",
        "count": 2,
        "require_word": True,
    }
    assert score_development_answer(row, 'There are 2 L characters in "yellow".')
    assert score_development_answer(
        row, 'y-e-l-l-o-w. There are 2 L characters in "yellow".'
    )
    for answer in (
        'There are 2 L characters in "yellor".',
        'There are 3 L characters in "yellow".',
        'y-e-l-l-o-r. There are 2 L characters in "yellow".',
        'There are 2 L characters in "yellow", or there are 3 L characters.',
        "2",
    ):
        assert not score_development_answer(row, answer)


def test_q3_aliases_accept_observed_paraphrases_without_weak_abbreviations():
    prompts = {
        "science fair": "volcano models",
        "retirement party": "retiring coworker",
        "proposal": "ring box",
        "Saint Patrick's Day": "shamrock",
    }
    for answer, clue in prompts.items():
        row = next(
            r
            for r in examples()
            if r["tag"] == "Q3" and clue in r["messages"][0]["content"]
        )
        assert score_development_answer(row, f"It is a {answer}.")
        assert not score_development_answer(row, "It is a wedding.")
        assert not score_development_answer(row, f"It is not a {answer}.")
    row = next(
        r
        for r in examples()
        if r["tag"] == "Q3" and "shamrock" in r["messages"][0]["content"]
    )
    assert not score_development_answer(row, "It is St.")


def test_q3_only_scope_preserves_locked_rows_and_split_membership():
    import hashlib

    lock = json.loads((ROOT / "assignment_6/q3_scope_lock.json").read_text())
    raw = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(raw) == 500
    assert sum(r["tag"] == "Q3" for r in raw) == 40
    training, development = stratified_family_split(examples())
    for tag, expected in lock["locked_groups"].items():
        group = [r for r in raw if r["tag"] == tag]
        payload = json.dumps(
            group, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        assert hashlib.sha256(payload.encode()).hexdigest() == expected["rows_sha256"]
        assert [r["id"] for r in group] == expected["dataset_ids_in_order"]
        assert [r["id"] for r in training if r["tag"] == tag] == expected[
            "training_ids"
        ]
        assert [r["id"] for r in development if r["tag"] == tag] == expected[
            "development_ids"
        ]
