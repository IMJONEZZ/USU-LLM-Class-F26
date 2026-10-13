import json

from src import config, data
from tests.helpers import (
    ROW,
    _run_pipeline,
    _teacher_chain,
    _train_row,
    _trow,
    _write_all,
)


def test_pipeline_train_data_requires_per_category(tmp_path):
    result = _run_pipeline("train-data", "--out", str(tmp_path / "t.jsonl"))
    assert result.returncode != 0
    assert "per-category" in result.stderr


def _chain_rows():
    rows = []
    for category in config.CATEGORIES:
        for i in range(3):
            rows.append(
                {**ROW, "id": f"{category}-{i}", "category": category, "chain": "c"}
            )
    return rows


def test_pipeline_sample_takes_per_category_rows_and_is_seeded(tmp_path):
    train = tmp_path / "train.jsonl"
    data.write_jsonl(train, _chain_rows())
    outs = []
    for name in ("a.json", "b.json"):
        out = tmp_path / name
        result = _run_pipeline(
            "sample",
            "--train",
            str(train),
            "--per-category",
            "2",
            "--seed",
            "0",
            "--out",
            str(out),
        )
        assert result.returncode == 0, result.stderr
        outs.append(out.read_text(encoding="utf-8"))
    picked = json.loads(outs[0])
    assert len(picked) == 2 * len(config.CATEGORIES)
    assert {r["category"] for r in picked} == set(config.CATEGORIES)
    assert outs[0] == outs[1]


def test_pipeline_sample_missing_train_file_fails_with_a_clear_message(tmp_path):
    result = _run_pipeline(
        "sample",
        "--train",
        str(tmp_path / "nope.jsonl"),
        "--per-category",
        "2",
        "--out",
        str(tmp_path / "o.json"),
    )
    assert result.returncode != 0
    assert "train-data" in result.stderr


def test_pipeline_test_data_records_the_hash(tmp_path):
    test_path, _, _, hash_path = _write_all(tmp_path, [_trow(1), _trow(2)], [])
    result = _run_pipeline(
        "test-data", "--test-set", str(test_path), "--hash", str(hash_path)
    )
    assert result.returncode == 0, result.stderr
    (record,) = data.read_jsonl(hash_path)
    assert record["sha256"] == data.file_sha256(test_path)
    assert "2 rows" in result.stdout


def test_pipeline_test_data_failure_names_the_row_and_records_nothing(tmp_path):
    rows = [_trow(1), _trow(2, chain=_teacher_chain("12"))]
    test_path, _, _, hash_path = _write_all(tmp_path, rows, [])
    result = _run_pipeline(
        "test-data", "--test-set", str(test_path), "--hash", str(hash_path)
    )
    assert result.returncode != 0
    assert "t-2" in result.stderr
    assert not hash_path.exists()


def test_pipeline_validate_fails_on_leakage_naming_the_test_row(tmp_path):
    train = [_train_row(1, question="Hand written question number 1?")]
    test_path, train_path, eval_path, hash_path = _write_all(
        tmp_path, [_trow(1)], train
    )
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode != 0
    assert "t-1" in result.stderr


def test_pipeline_validate_warns_when_the_test_set_changed(tmp_path):
    test_path, train_path, eval_path, hash_path = _write_all(
        tmp_path, [_trow(1)], [_train_row(1)]
    )
    data.record_test_set_hash(test_path, hash_path)
    data.write_jsonl(test_path, [_trow(1), _trow(2)])
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "changed" in result.stderr


def test_pipeline_validate_missing_train_file_fails_clearly(tmp_path):
    test_path, _, eval_path, hash_path = _write_all(tmp_path, [_trow(1)], [])
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(tmp_path / "nope.jsonl"),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode != 0
    assert "train-data" in result.stderr


# ---- M5: eval-data command ----


def test_pipeline_eval_data_invalid_test_set_fails_with_the_row_id(tmp_path):
    test_path = tmp_path / "test_set.jsonl"
    data.write_jsonl(test_path, [_trow(1, chain=_teacher_chain("12"))])
    result = _run_pipeline(
        "eval-data", "--test-set", str(test_path), "--out", str(tmp_path / "e.jsonl")
    )
    assert result.returncode != 0
    assert "t-1" in result.stderr
