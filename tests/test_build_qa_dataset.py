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


def test_get_and_download_falseqa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"question,answer\nx,y\n"

    monkeypatch.setattr(bq, "urlopen", lambda *a, **k: _Resp())
    assert bq._get("https://example.com/x") == b"question,answer\nx,y\n"

    monkeypatch.setattr(bq, "RAW", tmp_path)
    calls = {"n": 0}

    def fake_get(url: str) -> bytes:
        calls["n"] += 1
        name = url.rsplit("/", 1)[-1]
        if name == "train.csv" and "dataset" in url:
            return b"404: Not Found"
        if name == "dev.csv" and "dataset" in url:
            return b"<html>nope</html>"
        if name == "test.csv" and "dataset" in url:
            raise bq.URLError("skip")
        if name == "train.csv":
            return b"question,answer,label\nq?,a,0\n"
        raise bq.HTTPError(url, 404, "x", hdrs=None, fp=None)

    monkeypatch.setattr(bq, "_get", fake_get)
    out = bq.download_falseqa()
    assert out == tmp_path / "falseqa"
    assert (out / "train.csv").exists()
    # Seed remaining candidate filenames so the cache path skips every URL.
    for name in ("dev.csv", "test.csv"):
        (out / name).write_text("question,answer,label\nz,z,0\n", encoding="utf-8")
    before = calls["n"]
    bq.download_falseqa()
    assert calls["n"] == before

    monkeypatch.setattr(bq, "_get", lambda url: (_ for _ in ()).throw(bq.URLError("x")))
    monkeypatch.setattr(bq, "RAW", tmp_path / "raw2")
    with pytest.raises(RuntimeError, match="Could not download FalseQA"):
        bq.download_falseqa()


def test_download_brainteaser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bq, "RAW", tmp_path)

    def bad_get(url: str) -> bytes:
        raise OSError("offline")

    monkeypatch.setattr(bq, "_get", bad_get)
    with pytest.raises(RuntimeError, match="Could not download BrainTeaser"):
        bq.download_brainteaser()

    monkeypatch.setattr(bq, "_get", lambda url: b"PK" + b"\0" * 1200)

    class _Zip:
        pwd = None

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extractall(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "ok.txt").write_text("x", encoding="utf-8")

    import sys
    from types import ModuleType

    fake = ModuleType("pyzipper")
    fake.AESZipFile = lambda *a, **k: _Zip()
    monkeypatch.setitem(sys.modules, "pyzipper", fake)
    extracted = bq.download_brainteaser()
    assert (extracted / ".extracted_ok").exists()
    # Second call hits cache.
    assert bq.download_brainteaser() == extracted

    # Missing pyzipper path.
    marker = extracted / ".extracted_ok"
    marker.unlink()
    monkeypatch.setitem(sys.modules, "pyzipper", None)
    import builtins

    real_import = builtins.__import__

    def no_pyzipper(name, *args, **kwargs):
        if name == "pyzipper":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pyzipper)
    with pytest.raises(RuntimeError, match="install pyzipper"):
        bq.download_brainteaser()


def test_download_hf_datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bq, "RAW", tmp_path)

    class _Table(list):
        pass

    fake_ds = {"train": _Table([{"question": "Q?", "answer": False}])}

    def fake_load_dataset(name, *a, **k):
        return fake_ds

    import sys
    from types import ModuleType

    datasets = ModuleType("datasets")
    datasets.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    strat = bq.download_strategyqa()
    assert (strat / "train.jsonl").exists()
    truth = bq.download_truthfulqa()
    assert (truth / "train.jsonl").exists()


def test_extract_edge_cases_and_npy(tmp_path: Path):
    false_root = tmp_path / "falseqa"
    false_root.mkdir()
    (false_root / "blank.csv").write_text(
        "question,answer\n,skipme\nOk?,yes\n", encoding="utf-8"
    )
    assert len(bq._extract_questions_falseqa(false_root)) == 1

    strat = tmp_path / "strategyqa"
    strat.mkdir()
    (strat / "t.jsonl").write_text(
        json.dumps({"qid": "1", "question": "  "}) + "\n", encoding="utf-8"
    )
    assert bq._extract_questions_strategyqa(strat) == []

    truth = tmp_path / "truthfulqa"
    truth.mkdir()
    (truth / "t.jsonl").write_text(
        json.dumps({"question": "", "best_answer": "x"}) + "\n", encoding="utf-8"
    )
    assert bq._extract_questions_truthfulqa(truth) == []

    brain = tmp_path / "brain"
    brain.mkdir()
    (brain / ".hidden.json").write_text("[]", encoding="utf-8")
    (brain / "bad.json").write_text("{not json", encoding="utf-8")
    np.save(
        brain / "q.npy", np.array(["Is this npy puzzle long enough?"], dtype=object)
    )
    (brain / "dup.json").write_text(
        json.dumps(["Same question here?", "same question here?"]),
        encoding="utf-8",
    )
    rows = bq._extract_questions_brainteaser(brain)
    assert any("npy puzzle" in r["question"] for r in rows)
    assert sum(1 for r in rows if "same question here" in r["question"].lower()) == 1


def test_first_sentence_empty_and_long_report(tmp_path: Path):
    assert bq._first_sentence("   ") == ""
    assert bq._first_sentence("Wow! More.") == "Wow!"
    long_q = "Q" * 250
    path = tmp_path / "long.md"
    bq.write_leakage_report(
        [
            {
                "dataset": "x",
                "example_id": "e",
                "question": long_q,
                "match_type": "exact",
                "stem_index": 1,
                "graded_stem": "stem",
                "score": 1.0,
                "scenario": "opera_people",
                "action": "excluded",
            }
        ],
        {"x": 1},
        path,
    )
    text = path.read_text(encoding="utf-8")
    assert "..." in text
    assert "scenario=`opera_people`" in text


def test_cmd_download_inspect_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bq, "download_falseqa", lambda: tmp_path / "f")
    monkeypatch.setattr(bq, "download_brainteaser", lambda: tmp_path / "b")
    monkeypatch.setattr(bq, "download_strategyqa", lambda: tmp_path / "s")
    monkeypatch.setattr(bq, "download_truthfulqa", lambda: tmp_path / "t")
    monkeypatch.setattr(bq, "RAW", tmp_path)
    bq.cmd_download(type("A", (), {})())

    monkeypatch.setattr(bq, "DATA", tmp_path)
    monkeypatch.setattr(bq, "collect_all_questions", list)
    with pytest.raises(SystemExit, match="No questions found"):
        bq.cmd_inspect(type("A", (), {})())

    monkeypatch.setattr(
        bq,
        "collect_all_questions",
        lambda: [
            {
                "dataset": "falseqa",
                "example_id": "f:1",
                "question": GRADED_QUESTIONS[0],
            }
        ],
    )
    bq.cmd_inspect(type("A", (), {})())
    assert (tmp_path / "leakage_report.md").exists()
    assert (tmp_path / "leakage_findings.json").exists()

    assert bq._excluded_ids()  # findings exist
    monkeypatch.setattr(bq, "DATA", tmp_path / "empty_data")
    assert bq._excluded_ids() == set()

    called = []
    monkeypatch.setattr(bq, "cmd_download", lambda a: called.append("d"))
    monkeypatch.setattr(bq, "cmd_inspect", lambda a: called.append("i"))
    monkeypatch.setattr(bq, "cmd_convert", lambda a: called.append("c"))
    bq.main(["download"])
    bq.main(["inspect"])
    bq.main(["phase1"])
    bq.main(["convert", "--seed", "1"])
    assert called == ["d", "i", "d", "i", "c"]

    import runpy
    import sys

    exits = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exits.append(code))
    monkeypatch.setattr(sys, "argv", [bq.__file__, "download"])
    runpy.run_path(bq.__file__, run_name="__main__")
    assert "d" in called


def test_convert_exclusion_and_absent_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(bq, "RAW", tmp_path)
    monkeypatch.setattr(bq, "QA", tmp_path / "qa")
    monkeypatch.setattr(bq, "DATA", tmp_path)
    (tmp_path / "leakage_findings.json").write_text(
        json.dumps([{"example_id": "falseqa:train:0", "action": "excluded"}]),
        encoding="utf-8",
    )
    (tmp_path / "falseqa").mkdir()
    # Row 0 excluded by id; row 1 excluded by graded exact text (714-715).
    (tmp_path / "falseqa" / "train.csv").write_text(
        "question,answer,label\n"
        "Odd premise only?,No.,1\n"
        f"{GRADED_QUESTIONS[1]},No.,0\n"
        "Keep this?,Yes.,0\n",
        encoding="utf-8",
    )
    (tmp_path / "brainteaser" / "extracted").mkdir(parents=True)
    np.save(
        tmp_path / "brainteaser" / "extracted" / "SP_train.npy",
        np.array(["not-a-dict", {"question": "", "answer": "x"}], dtype=object),
        allow_pickle=True,
    )
    (tmp_path / "strategyqa").mkdir()
    (tmp_path / "strategyqa" / "t.jsonl").write_text(
        json.dumps({"qid": "1", "question": "Q?", "answer": False, "facts": ""})
        + "\n"
        + json.dumps({"qid": "2", "question": "Q2?", "answer": "maybe"})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "truthfulqa").mkdir()

    # Inject excluded id + graded stem to trip both absent_ok branches.
    real_write = bq._write_jsonl

    def inject(path: Path, rows: list[dict]) -> None:
        if path.name == "pending_train.jsonl":
            rows = list(rows) + [
                {
                    "example_id": "falseqa:train:0",
                    "instruction": "should have been excluded by id",
                    "response": "x",
                    "category": "c",
                    "source": "s",
                    "review_status": "pending",
                },
                {
                    "example_id": "leak",
                    "instruction": GRADED_QUESTIONS[2],
                    "response": "x",
                    "category": "c",
                    "source": "s",
                    "review_status": "pending",
                },
            ]
        real_write(path, rows)

    monkeypatch.setattr(bq, "_write_jsonl", inject)
    with pytest.raises(SystemExit, match="Graded exact matches still present"):
        bq.cmd_convert(type("A", (), {"seed": 0})())
