"""Tests for the frozen runner contract and predeclared selection policy."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "assignment_6/candidates/q8_v5_entity"
spec = importlib.util.spec_from_file_location("q8_runtime", HERE / "runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_runtime_uses_exact_manifest_and_rejects_setting_changes():
    rows, training, development, recipe = runtime.load_locked_candidate()
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    assert len(rows) == 522
    assert training == manifest["training"]
    assert development == manifest["development"]
    runtime.assert_recipe_settings(recipe["settings"], recipe)
    changed = dict(recipe["settings"], seed=43)
    with pytest.raises(ValueError, match="settings differ"):
        runtime.assert_recipe_settings(changed, recipe)


def fixtures():
    ids = json.loads((HERE / "development_spec.json").read_text())["rows"]
    records = [
        {
            "step": step,
            "epoch": index + 1,
            "development_loss": loss,
            "metrics": {f"eval_skill_match_Q{n}": 0.5 for n in range(1, 11)},
        }
        for index, (step, loss) in enumerate([(54, 1.0), (108, 0.8), (162, 0.7)])
    ]
    review = {
        "policy_sha256": runtime.file_sha(HERE / "selection_policy.json"),
        "checkpoints": {
            str(step): [
                {"id": key, "status": "partial", "reason": "missing distinction"}
                for key in ids
            ]
            for step in (54, 108, 162)
        },
    }
    return records, review


def test_selection_replaces_only_q8_and_uses_loss_tiebreak():
    records, review = fixtures()
    selection = runtime.choose_checkpoint(records, review)
    assert selection["selected_step"] == 162
    assert selection["checkpoint_comparison"][0]["selection_macro"] == 0.45
    # A genuine Q8 gain can beat the lower loss; partial matches give no credit.
    for row in review["checkpoints"]["108"]:
        row.update(status="complete", reason="all applicable criteria satisfied")
    selection = runtime.choose_checkpoint(records, review)
    assert selection["selected_step"] == 108


def test_selection_rejects_missing_cases_and_wrong_policy():
    records, review = fixtures()
    missing = copy.deepcopy(review)
    missing["checkpoints"]["54"].pop()
    with pytest.raises(ValueError, match="eight frozen"):
        runtime.choose_checkpoint(records, missing)
    wrong = dict(review, policy_sha256="wrong")
    with pytest.raises(ValueError, match="wrong frozen policy"):
        runtime.choose_checkpoint(records, wrong)
    with pytest.raises(ValueError, match="all three"):
        runtime.choose_checkpoint(records[:2], review)
