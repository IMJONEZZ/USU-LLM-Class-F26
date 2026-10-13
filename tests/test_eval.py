import json

import numpy as np
import pytest

from src import config, data
from src import eval as evaluation
from tests.helpers import (
    _grade_row,
    _HFStyleTokenizer,
    _teacher_chain,
    _tiny_tokenizer,
    _trow,
    _write_test_set,
)


# ---- M7: cut_at_final_answer ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_teacher_chain("9") + "\n\nQuestion: What next?\nRestate: more", _teacher_chain("9")),
        ("Restate: x\nFINAL ANSWER: 9\nand then rambling", "Restate: x\nFINAL ANSWER: 9"),
        ("Restate: x\nFinal   answer : 9\n### Instruction:\nfoo", "Restate: x\nFinal   answer : 9"),
        ("Final answer: 9", "Final answer: 9"),
        ("Final answer: 9\nFinal answer: 10", "Final answer: 9"),
        ("Final answer: 9   \nmore", "Final answer: 9"),
        ("Restate: the Final answer: is tricky\nSolve: s\nFinal answer: 9\nextra", "Restate: the Final answer: is tricky\nSolve: s\nFinal answer: 9"),
        ("Restate: no label here   \n", "Restate: no label here"),
        ("", ""),
    ],
)  # fmt: skip
def test_cut_at_final_answer(text, expected):
    assert evaluation.cut_at_final_answer(text) == expected


def test_grade_a_correct_well_formed_answer():
    graded = evaluation.grade(_grade_row(), _teacher_chain("9"))
    assert (graded["correct"], graded["format_ok"], graded["final_answer"]) == (
        True,
        True,
        "9",
    )
    assert graded["id"] == "t-1"
    assert graded["category"] == "lateral_thinking"


def test_grade_a_wrong_but_well_formed_answer():
    graded = evaluation.grade(_grade_row(), _teacher_chain("12"))
    assert (graded["correct"], graded["format_ok"]) == (False, True)


def test_grade_accepts_an_alternative_answer():
    assert evaluation.grade(_grade_row(), _teacher_chain("nine"))["correct"] is True


def test_grade_output_cut_off_before_the_final_answer_is_wrong_and_a_format_failure():
    truncated = "\n".join(_teacher_chain("9").split("\n")[:3])
    graded = evaluation.grade(_grade_row(), truncated)
    assert graded["final_answer"] is None
    assert (graded["correct"], graded["format_ok"]) == (False, False)


def test_grade_right_answer_with_a_broken_format_counts_as_correct_but_not_compliant():
    broken = "Final answer: 9"
    graded = evaluation.grade(_grade_row(), broken)
    assert (graded["correct"], graded["format_ok"]) == (True, False)


@pytest.mark.parametrize("output", ["", "   ", "no label anywhere", "Final answer:"])
def test_grade_unparsable_output_is_wrong_and_not_compliant(output):
    graded = evaluation.grade(_grade_row(), output)
    assert (graded["correct"], graded["format_ok"]) == (False, False)


def test_grade_records_the_response_length():
    output = _teacher_chain("9")
    assert evaluation.grade(_grade_row(), output)["length"] == len(output)


def test_grade_keeps_the_source_id_of_an_eval_variant():
    variant = _grade_row(id="t-1-v0", source_id="t-1")
    assert evaluation.grade(variant, _teacher_chain("9"))["source_id"] == "t-1"


# ---- M7: score ----
def _rows_by_category(counts):
    rows = []
    for category, n in counts.items():
        for i in range(n):
            rows.append(_grade_row(id=f"{category}-{i}", category=category))
    return rows


def test_score_of_no_rows_is_undefined_everywhere_not_zero():
    metrics = evaluation.score([], [])
    assert metrics["n"] == 0
    assert metrics["accuracy"] == "undefined"
    assert metrics["format_compliance"] == "undefined"
    assert metrics["avg_response_chars"] == "undefined"
    assert all(v["accuracy"] == "undefined" for v in metrics["per_category"].values())


def test_score_a_category_with_no_rows_reports_undefined_not_zero():
    rows = _rows_by_category({"false_premise": 2})
    metrics = evaluation.score(rows, [_teacher_chain("9"), _teacher_chain("12")])
    assert metrics["per_category"]["false_premise"] == {
        "n": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert metrics["per_category"]["time_sensitive"]["accuracy"] == "undefined"
    assert metrics["per_category"]["time_sensitive"]["n"] == 0


def test_score_overall_numbers_and_category_order():
    rows = _rows_by_category({"false_premise": 2, "misconception": 2})
    outputs = [
        _teacher_chain("9"),
        _teacher_chain("9"),
        _teacher_chain("12"),
        "garbage",
    ]
    metrics = evaluation.score(rows, outputs)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["format_compliance"] == 0.75
    assert metrics["avg_response_chars"] == pytest.approx(
        sum(len(o) for o in outputs) / 4
    )
    assert list(metrics["per_category"]) == config.CATEGORIES


def test_score_counts_unparsable_outputs_as_wrong():
    rows = _rows_by_category({"misconception": 2})
    metrics = evaluation.score(rows, ["", "nothing"])
    assert metrics["accuracy"] == 0.0
    assert metrics["per_category"]["misconception"]["correct"] == 0


def test_score_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="2 rows but 1 outputs"):
        evaluation.score(_rows_by_category({"misconception": 2}), [_teacher_chain("9")])


def test_score_a_category_outside_the_known_list_raises():
    rows = [_grade_row(category="character_counting")]
    with pytest.raises(ValueError, match="character_counting"):
        evaluation.score(rows, [_teacher_chain("9")])


# ---- M7: evaluate ----
def test_evaluate_builds_one_prompt_per_row_and_scores_the_outputs():
    rows = _rows_by_category({"false_premise": 1, "misconception": 1})
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return [_teacher_chain("9"), _teacher_chain("12")]

    results, metrics = evaluation.evaluate(
        rows, generate, lambda row: f"PROMPT {row['id']}"
    )
    assert seen == ["PROMPT false_premise-0", "PROMPT misconception-0"]
    assert [r["correct"] for r in results] == [True, False]
    assert metrics["accuracy"] == 0.5


def test_evaluate_with_no_rows_makes_no_generation_call():
    calls = []
    results, metrics = evaluation.evaluate(
        [], lambda prompts: calls.append(prompts) or [], lambda row: "p"
    )
    assert (results, calls) == ([], [])
    assert metrics["accuracy"] == "undefined"


def test_evaluate_a_generator_that_returns_the_wrong_number_of_outputs_raises():
    rows = _rows_by_category({"misconception": 2})
    with pytest.raises(ValueError, match="2 rows but 1 outputs"):
        evaluation.evaluate(rows, lambda prompts: ["only one"], lambda row: "p")


# ---- M7: few-shot examples and prompts (training data only) ----


def test_finetuned_prompt_is_the_plain_training_prompt():
    row = _grade_row(question="Just the question please?")
    assert evaluation.finetuned_prompt(row) == data.format_prompt(
        "Just the question please?"
    )


def test_check_prompt_length_accepts_prompts_at_the_limit():
    tok = _tiny_tokenizer()
    prompt = "a b c"
    limit = len(tok.encode(prompt, add_special_tokens=True).ids)
    assert evaluation.check_prompt_length([prompt], tok, limit) is None


def test_check_prompt_length_names_the_longest_prompt_over_the_limit():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="2 prompts") as caught:
        evaluation.check_prompt_length(
            ["a b", "a b c d e f g", "a b c d e f g h"], tok, 6
        )
    assert "longest" in str(caught.value)


COUNTS = {c: 5 for c in config.CATEGORIES}


def test_weak_categories_all_equal_flags_nothing():
    accuracy = {c: 0.8 for c in config.CATEGORIES}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.1) == []


def test_weak_categories_one_eligible_category_has_nothing_to_compare_against():
    assert (
        evaluation.find_weak_categories(
            {"misconception": 0.1}, {"misconception": 5}, gap=0.1
        )
        == []
    )


def test_weak_categories_flags_a_category_that_lags_the_average():
    accuracy = {"false_premise": 0.9, "misconception": 0.9, "pun_ambiguity": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "pun_ambiguity"
    ]


def test_weak_categories_a_gap_exactly_at_the_threshold_is_flagged():
    accuracy = {"false_premise": 1.0, "misconception": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.25) == [
        "misconception"
    ]


def test_weak_categories_a_gap_just_under_the_threshold_is_not_flagged():
    accuracy = {"false_premise": 1.0, "misconception": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.26) == []


def test_weak_categories_ties_are_ordered_by_accuracy_then_name():
    accuracy = {
        "false_premise": 1.0,
        "misconception": 0.4,
        "pun_ambiguity": 0.4,
        "lateral_thinking": 0.2,
        "time_sensitive": 1.0,
    }
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "lateral_thinking",
        "misconception",
        "pun_ambiguity",
    ]


def test_weak_categories_below_the_minimum_source_count_are_not_flagged_or_averaged():
    accuracy = {"false_premise": 0.9, "misconception": 0.9, "pun_ambiguity": 0.0}
    counts = {"false_premise": 5, "misconception": 5, "pun_ambiguity": 2}
    assert evaluation.find_weak_categories(accuracy, counts, gap=0.1) == []


def test_weak_categories_ignore_nan_and_undefined_accuracies():
    accuracy = {
        "false_premise": 0.9,
        "misconception": float("nan"),
        "pun_ambiguity": "undefined",
        "lateral_thinking": 0.3,
    }
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "lateral_thinking"
    ]


def test_weak_categories_a_category_missing_from_the_counts_is_not_eligible():
    accuracy = {"false_premise": 0.9, "misconception": 0.1}
    assert (
        evaluation.find_weak_categories(accuracy, {"false_premise": 5}, gap=0.1) == []
    )


@pytest.mark.parametrize("gap", [0, -0.1])
def test_weak_categories_needs_a_positive_gap(gap):
    with pytest.raises(ValueError, match="gap"):
        evaluation.find_weak_categories({"misconception": 0.5}, COUNTS, gap=gap)


# ---- M7: timer, versions, GPU summary, result record ----
def test_timer_records_the_elapsed_time(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(evaluation, "_now", lambda: next(ticks))
    with evaluation.Timer() as timer:
        pass
    assert timer.seconds == 2.5


def test_timer_records_the_time_even_if_the_block_raises(monkeypatch):
    ticks = iter([1.0, 4.0])
    monkeypatch.setattr(evaluation, "_now", lambda: next(ticks))
    timer = evaluation.Timer()
    with pytest.raises(RuntimeError, match="boom"), timer:
        raise RuntimeError("boom")
    assert timer.seconds == 3.0


def test_package_versions_reports_installed_missing_and_python():
    versions = evaluation.package_versions(["numpy", "definitely-not-a-package-xyz"])
    assert versions["numpy"] == np.__version__
    assert versions["definitely-not-a-package-xyz"] is None
    assert versions["python"].count(".") >= 1


def test_gpu_summary_has_none_fields_when_there_is_no_gpu():
    def no_smi():
        raise FileNotFoundError("nvidia-smi")

    with evaluation.GpuMonitor(run_smi=no_smi, torch_peak_mib=lambda: None) as monitor:
        pass
    assert monitor.summary() == {
        "gpu_name": None,
        "peak_memory_mib": None,
        "mean_utilization_percent": None,
        "peak_torch_memory_allocated_mib": None,
    }


def test_gpu_summary_takes_the_peak_memory_and_mean_utilization_over_the_samples():
    readings = iter(
        ["NVIDIA L4, 1000, 40\n", "NVIDIA L4, 2000, 80\n", "NVIDIA L4, 1500, 60\n"]
    )
    monitor = evaluation.GpuMonitor(
        run_smi=lambda: next(readings), torch_peak_mib=lambda: 1234.5
    )
    for _ in range(3):
        monitor.sample_once()
    assert monitor.summary() == {
        "gpu_name": "NVIDIA L4",
        "peak_memory_mib": 2000,
        "mean_utilization_percent": 60.0,
        "peak_torch_memory_allocated_mib": 1234.5,
    }


def test_gpu_monitor_ignores_unreadable_smi_output():
    outputs = iter(["garbage", "", "NVIDIA L4, 500, 10"])
    monitor = evaluation.GpuMonitor(
        run_smi=lambda: next(outputs), torch_peak_mib=lambda: None
    )
    for _ in range(3):
        monitor.sample_once()
    assert monitor.summary()["peak_memory_mib"] == 500


def test_gpu_monitor_samples_on_entry_and_exit():
    count = []

    def smi():
        count.append(1)
        return "NVIDIA L4, 100, 5"

    with evaluation.GpuMonitor(run_smi=smi, torch_peak_mib=lambda: None, interval=10):
        pass
    assert len(count) >= 2


def _result(**overrides):
    base = {
        "split": "eval",
        "metrics": evaluation.score([], []),
        "rows": [_grade_row()],
        "batch_size": 8,
        "max_seq_length": 2048,
        "versions": {"python": "3.14"},
        "eval_seconds": 12.0,
        "gpu_summary": {"gpu_name": None},
        "prompt_style": "zero_shot",
        "max_new_tokens": 384,
    }
    return evaluation.build_result(**{**base, **overrides})


def test_build_result_has_every_required_field_and_is_json_serializable():
    result = _result()
    assert all(field in result for field in evaluation.REQUIRED_RESULT_FIELDS)
    assert result["model_name"] == config.MODEL_NAME
    assert result["test_size"] == 1
    assert json.loads(json.dumps(result)) == result


def test_build_result_records_the_split_prompt_style_and_generation_settings():
    result = _result()
    assert (result["split"], result["prompt_style"]) == ("eval", "zero_shot")
    assert result["max_new_tokens"] == 384
    assert result["gpu_type"] == config.GPU_TYPE
    assert "fewshot_ids" not in result


def test_save_result_writes_json_atomically(tmp_path):
    path = tmp_path / "results" / "baseline.json"
    evaluation.save_result(path, _result())
    assert json.loads(path.read_text(encoding="utf-8"))["split"] == "eval"
    assert list(path.parent.iterdir()) == [path]


# ---- M7: the final test is run once ----
def test_the_test_split_is_refused_a_second_time_without_force(tmp_path):
    path = tmp_path / "final_test.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        evaluation.ensure_split_allowed("test", path, force=False)


def test_the_test_split_is_allowed_the_first_time_and_with_force(tmp_path):
    path = tmp_path / "final_test.json"
    assert evaluation.ensure_split_allowed("test", path, force=False) is None
    path.write_text("{}", encoding="utf-8")
    assert evaluation.ensure_split_allowed("test", path, force=True) is None


def test_the_eval_split_may_be_rerun(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{}", encoding="utf-8")
    assert evaluation.ensure_split_allowed("eval", path, force=False) is None


def test_an_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        evaluation.ensure_split_allowed("train", tmp_path / "x.json", force=False)


# ---- M7: the baseline prompt is zero-shot (a written instruction, no examples) ----
def test_baseline_prompt_has_the_instruction_the_question_and_the_response_marker():
    prompt = evaluation.baseline_prompt(
        _grade_row(question="A brand new question here?")
    )
    assert prompt.startswith(config.INSTRUCTION_MARKER)
    assert prompt.endswith(config.RESPONSE_MARKER)
    assert "A brand new question here?" in prompt
    assert all(label in prompt for label in config.SECTION_LABELS)


def test_baseline_prompt_contains_no_examples():
    prompt = evaluation.baseline_prompt(
        _grade_row(question="A brand new question here?")
    )
    assert prompt.count(config.RESPONSE_MARKER) == 1
    assert prompt.count(config.INSTRUCTION_MARKER) == 1


def test_baseline_prompt_is_the_same_for_the_same_question_and_differs_from_the_trained_prompt():
    row = _grade_row(question="Same question each time?")
    assert evaluation.baseline_prompt(row) == evaluation.baseline_prompt(dict(row))
    assert evaluation.baseline_prompt(row) != evaluation.finetuned_prompt(row)


def test_baseline_prompt_refuses_a_question_containing_a_template_marker():
    with pytest.raises(ValueError):
        evaluation.baseline_prompt(
            _grade_row(question=f"What is {config.RESPONSE_MARKER} this?")
        )


# ---- M7: where a run writes and which rows each split reads ----
def test_default_output_path_for_the_baseline_on_the_eval_split():
    assert evaluation.default_output_path("eval", "baseline") == config.BASELINE_PATH


def test_default_output_path_for_the_final_test_with_the_trained_model():
    assert evaluation.default_output_path("test", "finetuned") == config.FINAL_TEST_PATH


def test_an_explicit_output_always_wins(tmp_path):
    chosen = tmp_path / "mine.json"
    assert evaluation.default_output_path("test", "baseline", str(chosen)) == chosen


@pytest.mark.parametrize(
    ("split", "style"), [("eval", "finetuned"), ("test", "baseline")]
)
def test_other_combinations_must_name_their_output(split, style):
    with pytest.raises(ValueError, match="--output"):
        evaluation.default_output_path(split, style)


def test_load_split_rows_reads_the_eval_file(monkeypatch, tmp_path):
    path = tmp_path / "eval.jsonl"
    data.write_jsonl(path, [_trow(1), _trow(2)])
    monkeypatch.setattr(config, "EVAL_PATH", path)
    assert [r["id"] for r in evaluation.load_split_rows("eval")] == ["t-1", "t-2"]


def test_load_split_rows_reads_the_validated_test_set(monkeypatch, tmp_path):
    path = _write_test_set(tmp_path, [_trow(1)])
    monkeypatch.setattr(config, "TEST_SET_PATH", path)
    monkeypatch.setattr(config, "EXCLUDED_TEST_IDS", frozenset())
    assert [r["id"] for r in evaluation.load_split_rows("test")] == ["t-1"]


def test_load_split_rows_refuses_an_invalid_test_set(monkeypatch, tmp_path):
    path = _write_test_set(tmp_path, [_trow(1, chain=_teacher_chain("12"))])
    monkeypatch.setattr(config, "TEST_SET_PATH", path)
    with pytest.raises(ValueError, match="t-1"):
        evaluation.load_split_rows("test")


def test_load_split_rows_with_an_empty_eval_file_raises(monkeypatch, tmp_path):
    path = tmp_path / "eval.jsonl"
    data.write_jsonl(path, [])
    monkeypatch.setattr(config, "EVAL_PATH", path)
    with pytest.raises(ValueError, match="no rows"):
        evaluation.load_split_rows("eval")


def test_load_split_rows_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split"):
        evaluation.load_split_rows("train")


def test_check_prompt_length_works_with_a_transformers_style_tokenizer():
    hf = _HFStyleTokenizer(_tiny_tokenizer())
    assert evaluation.check_prompt_length(["a b c"], hf, 99) is None
    with pytest.raises(ValueError, match="1 prompt"):
        evaluation.check_prompt_length(["a b c d e f g h"], hf, 3)


# ---- M9: regressions and source counts ----
def test_a_category_that_drops_after_a_round_is_flagged_as_a_regression():
    previous = {"false_premise": 0.9, "misconception": 0.8, "pun_ambiguity": 0.3}
    current = {"false_premise": 0.6, "misconception": 0.8, "pun_ambiguity": 0.7}
    assert evaluation.find_regressions(previous, current, drop=0.2) == ["false_premise"]


def test_a_drop_exactly_at_the_threshold_is_flagged_and_just_under_is_not():
    assert evaluation.find_regressions({"a": 1.0}, {"a": 0.75}, drop=0.25) == ["a"]
    assert evaluation.find_regressions({"a": 1.0}, {"a": 0.75}, drop=0.26) == []


def test_regressions_ignore_undefined_nan_and_missing_categories():
    previous = {"a": "undefined", "b": 0.9, "c": float("nan"), "d": 0.9}
    current = {"a": 0.1, "b": "undefined", "c": 0.1}
    assert evaluation.find_regressions(previous, current, drop=0.1) == []


def test_regressions_are_ordered_by_the_size_of_the_drop_then_name():
    previous = {"a": 0.9, "b": 0.9, "c": 0.9}
    current = {"a": 0.5, "b": 0.3, "c": 0.5}
    assert evaluation.find_regressions(previous, current, drop=0.2) == ["b", "a", "c"]


@pytest.mark.parametrize("drop", [0, -0.1])
def test_regressions_need_a_positive_drop(drop):
    with pytest.raises(ValueError, match="drop"):
        evaluation.find_regressions({"a": 1.0}, {"a": 0.5}, drop=drop)


def test_source_counts_counts_distinct_source_questions_per_category():
    rows = [
        {"category": "misconception", "source_id": "q1"},
        {"category": "misconception", "source_id": "q1"},
        {"category": "misconception", "source_id": "q2"},
        {"category": "pun_ambiguity", "id": "q3"},
    ]
    assert evaluation.source_counts(rows) == {"misconception": 2, "pun_ambiguity": 1}
