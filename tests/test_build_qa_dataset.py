"""Unit tests for Assignment 6 QA dataset helpers (no network downloads)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import build_qa_dataset as bq
from src.trainer import GRADED_QUESTIONS, normalize_text, validate_qa_record


def test_inspect_item_exact_match_only():
    stem = GRADED_QUESTIONS[8]
    hits = bq.inspect_item(
        {"dataset": "x", "example_id": "e1", "question": stem.upper()}
    )
    assert len(hits) == 1
    assert hits[0]["match_type"] == "exact"
    assert hits[0]["stem_index"] == 9
    assert hits[0]["action"] == "excluded"


def test_inspect_item_near_miss_and_blank():
    assert bq.inspect_item({"question": "How many Ls are in parallel roughly?"}) == []
    assert bq.inspect_item({"question": "   "}) == []
    assert bq.inspect_item({}) == []


def test_scenario_and_cue_helpers():
    assert bq._cue_hit("united states president", ("united states", "usa"))
    assert not bq._cue_hit("hello", ("united states",))
    christmas = (
        "a boy runs downstairs and sees a tree in the living room "
        "with boxes under the tree"
    )
    hits = bq.scenario_matches(christmas)
    assert any(h["scenario"] == "christmas_tree_boxes" for h in hits)
    assert not any(
        h["scenario"] == "parallel_letters"
        for h in bq.scenario_matches("the roads are parallel")
    )
    assert any(
        h["scenario"] == "parallel_letters"
        for h in bq.scenario_matches("how many letters in parallel")
    )


def test_first_sentence_and_qa_record():
    assert bq._first_sentence("Hello world. More text.") == "Hello world."
    assert bq._first_sentence("No period here") == "No period here"
    row = bq._qa_record(
        example_id="t:1",
        instruction="Q?",
        response="A.",
        category="test",
        source="unit",
    )
    assert row is not None
    validate_qa_record(row)
    assert row["review_status"] == "pending"
    assert (
        bq._qa_record(
            example_id="t:2",
            instruction=" ",
            response="A",
            category="c",
            source="s",
        )
        is None
    )


def test_dedupe_and_jsonl_roundtrip(tmp_path: Path):
    rows = [
        bq._qa_record(
            example_id="a",
            instruction="Same Q?",
            response="one",
            category="c",
            source="s",
        ),
        bq._qa_record(
            example_id="b",
            instruction="same q?",
            response="two",
            category="c",
            source="s",
        ),
        bq._qa_record(
            example_id="c",
            instruction="Other?",
            response="three",
            category="c",
            source="s",
        ),
    ]
    kept, dropped = bq._dedupe_records(rows)
    assert dropped == 1
    assert len(kept) == 2
    path = tmp_path / "rows.jsonl"
    bq._write_jsonl(path, kept)
    loaded = bq._load_jsonl(path)
    assert len(loaded) == 2
    assert loaded[0]["example_id"] == "a"


def test_brainteaser_rows_from_obj():
    obj = {
        "q": "Why is the sky blue?",
        "nested": ["Is this a puzzle?", {"text": "What walks at dawn?"}],
    }
    rows = bq._brainteaser_rows_from_obj(obj, prefix="root")
    questions = {r["question"] for r in rows}
    assert "Why is the sky blue?" in questions
    assert "Is this a puzzle?" in questions


def test_extract_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    false_root = tmp_path / "falseqa"
    false_root.mkdir()
    (false_root / "train.csv").write_text(
        "question,answer,label\nIs the sun cold?,No it is hot,0\n",
        encoding="utf-8",
    )
    fq = bq._extract_questions_falseqa(false_root)
    assert fq and fq[0]["dataset"] == "falseqa"

    strat = tmp_path / "strategyqa"
    strat.mkdir()
    (strat / "strategyqa_train.jsonl").write_text(
        json.dumps(
            {
                "qid": "q1",
                "question": "Can a plane fly underwater?",
                "answer": False,
                "facts": ["Planes need air."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sq = bq._extract_questions_strategyqa(strat)
    assert sq[0]["example_id"].startswith("strategyqa:")

    truth = tmp_path / "truthfulqa"
    truth.mkdir()
    (truth / "generation.jsonl").write_text(
        json.dumps(
            {
                "question": "Do vaccines cause magnetism?",
                "best_answer": "No.",
                "category": "Health",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tq = bq._extract_questions_truthfulqa(truth)
    assert tq[0]["dataset"] == "truthfulqa"

    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / "puzzles.json").write_text(
        json.dumps(["What has keys but no locks?"]),
        encoding="utf-8",
    )
    (brain / "more.jsonl").write_text(
        json.dumps({"question": "Why did the chicken cross?"}) + "\n",
        encoding="utf-8",
    )
    assert len(bq._extract_questions_brainteaser(brain)) >= 2

    monkeypatch.setattr(bq, "RAW", tmp_path)
    (tmp_path / "brainteaser" / "extracted").mkdir(parents=True)
    (tmp_path / "brainteaser" / "extracted" / "x.json").write_text(
        json.dumps(["Unique brainteaser question here?"]),
        encoding="utf-8",
    )
    all_q = bq.collect_all_questions()
    assert any(r["dataset"] == "brainteaser" for r in all_q)


def test_write_leakage_report_empty_and_findings(tmp_path: Path):
    empty = tmp_path / "empty.md"
    bq.write_leakage_report([], {"falseqa": 3}, empty)
    text = empty.read_text(encoding="utf-8")
    assert "No exact matches found" in text
    assert "falseqa" in text

    filled = tmp_path / "filled.md"
    bq.write_leakage_report(
        [
            {
                "dataset": "falseqa",
                "example_id": "f:1",
                "question": GRADED_QUESTIONS[0],
                "match_type": "exact",
                "stem_index": 1,
                "graded_stem": GRADED_QUESTIONS[0],
                "action": "excluded",
            }
        ],
        {"falseqa": 1},
        filled,
    )
    assert "exact" in filled.read_text(encoding="utf-8")


def test_convert_helpers_with_tmp_raw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bq, "RAW", tmp_path)

    extracted = tmp_path / "brainteaser" / "extracted"
    extracted.mkdir(parents=True)
    np.save(
        extracted / "SP_train.npy",
        np.array(
            [
                {
                    "id": "1",
                    "question": "What gets wetter the more it dries?",
                    "answer": "A towel",
                }
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )
    bt = bq.convert_brainteaser()
    assert bt and bt[0]["category"] == "lateral_thinking"
    validate_qa_record(bt[0])

    falseqa = tmp_path / "falseqa"
    falseqa.mkdir()
    (falseqa / "train.csv").write_text(
        "question,answer,label\n"
        "Do fish live in clouds?,No. Fish live in water.,1\n"
        "Is water wet?,Yes it is.,0\n",
        encoding="utf-8",
    )
    fq = bq.convert_falseqa()
    assert {r["category"] for r in fq} == {"false_premise", "true_premise"}
    assert len(fq) == 2

    strat = tmp_path / "strategyqa"
    strat.mkdir()
    (strat / "train.jsonl").write_text(
        json.dumps(
            {
                "qid": "abc",
                "question": "Would a metal boat sink in mercury?",
                "answer": True,
                "facts": "Mercury is dense. Metal boats float in water.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sq = bq.convert_strategyqa()
    assert sq[0]["category"] == "strategy_reasoning"
    assert sq[0]["response"].startswith("Yes")

    truth = tmp_path / "truthfulqa"
    truth.mkdir()
    (truth / "gen.jsonl").write_text(
        json.dumps(
            {
                "question": "Can you catch a cold from cold weather?",
                "best_answer": "No, colds come from viruses.",
                "category": "Health",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tq = bq.convert_truthfulqa()
    assert tq[0]["category"] == "misconception"


def test_excluded_ids_and_cmd_convert(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    findings = tmp_path / "leakage_findings.json"
    findings.write_text(
        json.dumps(
            [
                {
                    "example_id": "bad:1",
                    "match_type": "exact",
                    "action": "excluded",
                },
                {
                    "example_id": "soft:1",
                    "match_type": "jaccard",
                    "action": "review",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bq, "DATA", tmp_path)
    assert bq._excluded_ids() == {"bad:1"}

    monkeypatch.setattr(bq, "RAW", tmp_path)
    monkeypatch.setattr(bq, "QA", tmp_path / "qa")
    (tmp_path / "falseqa").mkdir()
    (tmp_path / "falseqa" / "train.csv").write_text(
        "question,answer,label\nOdd premise?,No.,1\n",
        encoding="utf-8",
    )
    (tmp_path / "brainteaser" / "extracted").mkdir(parents=True)
    (tmp_path / "strategyqa").mkdir()
    (tmp_path / "truthfulqa").mkdir()

    args = type("A", (), {"seed": 0, "dev_ratio": 0.5})()
    bq.cmd_convert(args)
    train = tmp_path / "qa" / "pending_train.jsonl"
    dev = tmp_path / "qa" / "pending_dev.jsonl"
    assert train.exists() and dev.exists()
    graded = {normalize_text(s) for s in GRADED_QUESTIONS}
    for path in (train, dev):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert normalize_text(row["instruction"]) not in graded


def test_write_bytes(tmp_path: Path):
    path = tmp_path / "nested" / "f.bin"
    bq._write_bytes(path, b"abc")
    assert path.read_bytes() == b"abc"
