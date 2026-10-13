import importlib.util
import json

from src import config
from tests.helpers import ROOT

SPEC = importlib.util.spec_from_file_location(
    "decide_round", ROOT / "scripts" / "decide_round.py"
)
decide_round = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(decide_round)


def _write_judged(results_dir, round_n, accuracy):
    # the same shape scripts/judge_outputs.py saves
    per_category = {
        category: {
            "n": 0 if value == "undefined" else 3,
            "correct": 0,
            "accuracy": value,
        }
        for category, value in accuracy.items()
    }
    path = results_dir / f"round{round_n}_judged.json"
    path.write_text(
        json.dumps({"metrics": {"per_category": per_category}, "rows": []}),
        encoding="utf-8",
    )


def _all(value, **changes):
    return {**{category: value for category in config.CATEGORIES}, **changes}


def _run(tmp_path, *args):
    return decide_round.main([*args, "--results-dir", str(tmp_path)])


def test_a_category_far_below_the_others_is_weak_and_the_decision_is_continue(
    tmp_path,
):
    _write_judged(tmp_path, 0, _all(0.9, pun_ambiguity=0.3))
    result = _run(tmp_path, "--round", "0", "--gap", "0.3")
    assert result["weak"] == ["pun_ambiguity"]
    assert result["decision"] == "continue"


def test_no_weak_category_means_stop(tmp_path):
    _write_judged(tmp_path, 0, _all(0.8))
    result = _run(tmp_path, "--round", "0", "--gap", "0.3")
    assert result["weak"] == []
    assert result["decision"] == "stop"


def test_a_category_with_undefined_accuracy_is_not_compared(tmp_path):
    _write_judged(tmp_path, 0, _all(0.9, time_sensitive="undefined"))
    result = _run(tmp_path, "--round", "0", "--gap", "0.3")
    assert result["weak"] == []


def test_the_decision_reads_the_judged_file_of_the_round_asked_for(tmp_path):
    _write_judged(tmp_path, 0, _all(0.9, pun_ambiguity=0.0))
    _write_judged(tmp_path, 1, _all(0.8))
    assert _run(tmp_path, "--round", "1", "--gap", "0.3")["decision"] == "stop"
    assert _run(tmp_path, "--round", "0", "--gap", "0.3")["decision"] == "continue"


def test_drop_lists_categories_that_fell_since_the_previous_round(tmp_path):
    _write_judged(tmp_path, 0, _all(0.8))
    _write_judged(tmp_path, 1, _all(0.8, misconception=0.4))
    result = _run(tmp_path, "--round", "1", "--gap", "0.5", "--drop", "0.3")
    assert result["regressed"] == ["misconception"]


def test_without_drop_or_on_round_zero_regressions_are_not_checked(tmp_path):
    _write_judged(tmp_path, 0, _all(0.8))
    _write_judged(tmp_path, 1, _all(0.2))
    assert _run(tmp_path, "--round", "1", "--gap", "0.5")["regressed"] is None
    assert (
        _run(tmp_path, "--round", "0", "--gap", "0.5", "--drop", "0.1")["regressed"]
        is None
    )


def test_the_result_is_printed(tmp_path, capsys):
    _write_judged(tmp_path, 0, _all(0.9, pun_ambiguity=0.3))
    _run(tmp_path, "--round", "0", "--gap", "0.3")
    printed = capsys.readouterr().out
    assert "continue" in printed
    assert "pun_ambiguity" in printed
