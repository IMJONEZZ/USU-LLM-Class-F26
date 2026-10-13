import importlib.util
import json

from src import data, teacher
from src import eval as evaluation
from tests.helpers import ROOT

SPEC = importlib.util.spec_from_file_location(
    "judge_outputs", ROOT / "scripts" / "judge_outputs.py"
)
judge_outputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judge_outputs)


def _eval_row(row_id, category):
    return {
        "id": row_id,
        "category": category,
        "question": f"question {row_id}?",
        "answer": "5 pounds",
        "accepted_alternatives": ["five pounds"],
    }


def _setup(monkeypatch, tmp_path, replies):
    rows = [_eval_row("a", "false_premise"), _eval_row("b", "misconception")]
    monkeypatch.setattr(evaluation, "load_split_rows", lambda split: rows)
    queue = list(replies)
    monkeypatch.setattr(teacher, "_generate", lambda prompt, model, t: queue.pop(0))
    outputs = tmp_path / "outputs.jsonl"
    # the same row shape the Modal evaluation writes, so the handoff is real
    data.write_jsonl(
        outputs,
        [
            {
                "id": "a",
                "category": "false_premise",
                "final_answer": "5 pounds of wood",
                "correct": False,
            },
            {
                "id": "b",
                "category": "misconception",
                "final_answer": "5,000 pounds",
                "correct": False,
            },
        ],
    )
    return outputs


def test_judging_a_saved_outputs_file_writes_verdicts_and_judged_metrics(
    monkeypatch, tmp_path
):
    outputs = _setup(monkeypatch, tmp_path, ['{"correct": true}', '{"correct": false}'])
    out = tmp_path / "judged.json"
    judge_outputs.main(
        ["--split", "eval", "--outputs", str(outputs), "--out", str(out)]
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["metrics"]["n"] == 2
    assert saved["metrics"]["accuracy"] == 0.5
    assert saved["metrics"]["per_category"]["false_premise"]["correct"] == 1
    assert [(r["id"], r["judge"]) for r in saved["rows"]] == [("a", True), ("b", False)]


def test_each_saved_row_keeps_the_answer_that_was_judged(monkeypatch, tmp_path):
    outputs = _setup(monkeypatch, tmp_path, ['{"correct": true}', '{"correct": false}'])
    out = tmp_path / "judged.json"
    judge_outputs.main(
        ["--split", "eval", "--outputs", str(outputs), "--out", str(out)]
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [r["final_answer"] for r in saved["rows"]] == [
        "5 pounds of wood",
        "5,000 pounds",
    ]


def test_the_judged_file_is_rounded_to_three_places(monkeypatch, tmp_path):
    rows = [_eval_row(i, "false_premise") for i in "abc"]
    monkeypatch.setattr(evaluation, "load_split_rows", lambda split: rows)
    replies = ['{"correct": true}', '{"correct": false}', '{"correct": false}']
    monkeypatch.setattr(teacher, "_generate", lambda prompt, model, t: replies.pop(0))
    outputs = tmp_path / "outputs.jsonl"
    data.write_jsonl(
        outputs,
        [
            {
                "id": i,
                "category": "false_premise",
                "final_answer": "x",
                "correct": False,
            }
            for i in "abc"
        ],
    )
    out = tmp_path / "judged.json"
    judge_outputs.main(
        ["--split", "eval", "--outputs", str(outputs), "--out", str(out)]
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["metrics"]["accuracy"] == 0.333
    assert saved["metrics"]["per_category"]["false_premise"]["accuracy"] == 0.333
