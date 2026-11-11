import json
from types import SimpleNamespace

import dspy
import pytest

from src import config, data, plots, prompt_optimization, prompts

TASK = {"key": 1, "prompt": "p", "instruction_id": "punctuation:no_comma", "kwargs": {}}


GOOD, BAD = "no commas here", "one, two"


PARSE_ERROR = dspy.utils.exceptions.AdapterParseError(
    "ChatAdapter", dspy.Signature("prompt -> response"), "raw reply"
)


def scripted(items):
    """A stand-in for `predict`: each call returns, or raises, the next item."""
    items = iter(items)

    def predict(prompt):
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(response=item)

    return predict


def test_make_student_points_at_the_server_with_cache_and_json_fallback_off(
    monkeypatch,
):
    monkeypatch.setenv("MODAL_URL", "https://example.test")
    monkeypatch.setenv("MODAL_ID", "id")
    monkeypatch.setenv("MODAL_SECRET", "secret")
    lm, predict = prompt_optimization.make_student()
    assert lm.model == "openai/" + config.SERVED_NAME
    assert lm.kwargs["api_base"] == "https://example.test/v1"
    assert lm.kwargs["headers"] == {"Modal-Key": "id", "Modal-Secret": "secret"}
    assert lm.kwargs["temperature"] == config.TEMPERATURE
    assert lm.kwargs["max_tokens"] == config.MAX_NEW_TOKENS
    assert lm.cache is False  # a cached repeat would make temperature 1.0 pointless
    assert dspy.settings.adapter.use_json_adapter_fallback is False
    assert predict.signature.instructions == prompts.ONE_LINER


def test_optimize_main_prints_both_replies_the_messages_and_the_actual_cost(
    tmp_path, monkeypatch, capsys
):
    entry = {
        "messages": [{"role": "system", "content": "the system text"}],
        "usage": {"completion_tokens": 7},
    }
    fake_lm = SimpleNamespace(history=[entry, entry])
    student = (fake_lm, scripted(["first reply", PARSE_ERROR]))
    monkeypatch.setattr(prompt_optimization, "make_student", lambda: student)
    monkeypatch.setattr(data, "load_examples", lambda: ([], [TASK]))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text('{"train": [1]}')
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    prompt_optimization.demo()
    out = capsys.readouterr().out
    assert "reply 1: 'first reply'" in out
    assert "PARSE ERROR, raw reply: raw reply" in out
    assert "the system text" in out
    assert "14 output tokens" in out


BASELINE_TASKS = [
    {
        "key": 3,
        "prompt": "b first",
        "instruction_id": "punctuation:no_comma",
        "kwargs": {},
    },
    {
        "key": 1,
        "prompt": "a first",
        "instruction_id": "change_case:english_lowercase",
        "kwargs": {},
    },
    {
        "key": 2,
        "prompt": "a second",
        "instruction_id": "change_case:english_lowercase",
        "kwargs": {},
    },
    {
        "key": 9,
        "prompt": "a test one",
        "instruction_id": "change_case:english_lowercase",
        "kwargs": {},
    },
]


BASELINE_SPLITS = {"train": [1, 2, 3], "validation": [], "test": [9]}


def fake_writer(reply):
    return lambda messages: [reply]


def test_write_baseline_uses_the_first_train_task_of_each_type_and_saves_it(tmp_path):
    path = tmp_path / "prompts.json"
    prompt_optimization.write_baseline(
        fake_writer("  Be exact.  "), BASELINE_TASKS, BASELINE_SPLITS, path
    )
    saved = json.loads(path.read_text())
    assert saved["baseline"] == "Be exact."
    assert saved["writer"]["example_keys"] == [1, 3]
    assert "a first" in saved["writer"]["meta_prompt"]
    assert (
        "refrain from the use of any commas" in saved["writer"]["meta_prompt"]
    )  # the writer sees each rule
    assert (
        "a test one" not in saved["writer"]["meta_prompt"]
    )  # test tasks never reach the writer


def test_write_baseline_refuses_to_overwrite_a_frozen_baseline(tmp_path):
    path = tmp_path / "prompts.json"
    prompt_optimization.write_baseline(
        fake_writer("one"), BASELINE_TASKS, BASELINE_SPLITS, path
    )
    with pytest.raises(FileExistsError):
        prompt_optimization.write_baseline(
            fake_writer("two"), BASELINE_TASKS, BASELINE_SPLITS, path
        )
    assert json.loads(path.read_text())["baseline"] == "one"


def test_write_baseline_rejects_an_empty_reply_and_saves_nothing(tmp_path):
    path = tmp_path / "prompts.json"
    with pytest.raises(ValueError):
        prompt_optimization.write_baseline(
            fake_writer("  "), BASELINE_TASKS, BASELINE_SPLITS, path
        )
    assert not path.exists()


def test_metric_scores_the_parsed_response_and_accepts_gepas_extra_arguments():
    gold = SimpleNamespace(task=TASK)
    good = prompt_optimization.metric(
        gold, SimpleNamespace(response=GOOD), None, "predict", None
    )
    bad = prompt_optimization.metric(gold, SimpleNamespace(response=BAD))
    assert (good.score, bad.score) == (1.0, 0.0)
    assert "punctuation:no_comma" in bad.feedback


@pytest.mark.parametrize(
    "pred",
    [SimpleNamespace(response=""), SimpleNamespace(response=None), SimpleNamespace()],
)
def test_metric_gives_zero_and_feedback_for_an_empty_or_missing_response(pred):
    result = prompt_optimization.metric(SimpleNamespace(task=TASK), pred)
    assert result.score == 0.0
    assert "Failed instruction" in result.feedback


def test_gepa_examples_carry_the_task_and_only_the_prompt_as_input():
    tasks = [{**TASK, "key": 1}, {**TASK, "key": 2}]
    examples = prompt_optimization.gepa_examples(tasks, [2])
    assert len(examples) == 1
    assert examples[0].task["key"] == 2
    assert list(examples[0].inputs().keys()) == ["prompt"]


@pytest.mark.parametrize(
    "budget", [{}, {"auto": "light", "max_metric_calls": 5}, {"seed": 1}]
)
def test_build_gepa_needs_exactly_one_budget_setting(budget):
    with pytest.raises(ValueError):
        prompt_optimization.build_gepa(None, "logs", **budget)


def test_build_gepa_passes_the_decided_settings_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(dspy, "GEPA", lambda **kwargs: seen.update(kwargs))
    prompt_optimization.build_gepa("reflection", "logs", auto="light")
    assert seen["auto"] == "light"
    assert seen["add_format_failure_as_feedback"] is True
    assert seen["track_stats"] is True
    assert seen["seed"] == config.SEED
    assert seen["log_dir"] == "logs"
    assert seen["reflection_lm"] == "reflection"
    assert seen["metric"] is prompt_optimization.metric


def test_gepa_smoke_prints_the_candidates_and_what_gemini_was_shown(
    tmp_path, monkeypatch, capsys
):
    candidate = SimpleNamespace(
        signature=SimpleNamespace(instructions="a new instruction")
    )
    details = SimpleNamespace(
        candidates=[candidate],
        val_aggregate_scores=[0.5],
        best_idx=0,
        total_metric_calls=40,
    )
    reflection = SimpleNamespace(
        history=[{"messages": [{"content": "the reflection prompt"}], "cost": 0.01}]
    )
    student = SimpleNamespace(history=[{}] * 3)
    optimized = SimpleNamespace(detailed_results=details)
    monkeypatch.setattr(
        prompt_optimization,
        "run_gepa",
        lambda *a, **k: (optimized, student, reflection),
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"train": list(range(1, 13))}))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(data, "load_examples", lambda: ([], [TASK]))
    prompt_optimization.gepa_smoke()
    out = capsys.readouterr().out
    assert "candidate 0: a new instruction" in out
    assert "the reflection prompt" in out
    assert "3 student calls, 1 Gemini calls" in out


def fake_gepa_result():
    candidates = [
        SimpleNamespace(signature=SimpleNamespace(instructions="old")),
        SimpleNamespace(signature=SimpleNamespace(instructions="new")),
    ]
    details = SimpleNamespace(
        candidates=candidates,
        val_aggregate_scores=[0.4, 0.5555],
        best_idx=1,
        total_metric_calls=700,
        num_full_val_evals=3,
    )
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    reflection = SimpleNamespace(
        history=[{"usage": usage, "cost": 0.004}, {"usage": usage, "cost": 0.006}]
    )
    student = SimpleNamespace(history=[{}] * 7)
    optimized = SimpleNamespace(
        detailed_results=details, signature=SimpleNamespace(instructions="new")
    )
    return optimized, student, reflection


def test_summarize_gepa_sums_the_counters_across_both_models_and_records_the_seed():
    optimized, student, reflection = fake_gepa_result()
    record = prompt_optimization.summarize_gepa(
        optimized.detailed_results, student, reflection, 100.0, "logs"
    )
    assert record["seed"] == config.SEED
    assert record["student"]["calls"] == 7
    assert record["student"]["dollars"] == round(100 * config.T4_PER_SECOND, 2)
    assert record["gemini"] == {
        "model": config.GEMINI_MODEL,
        "calls": 2,
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "dollars": 0.01,
    }
    assert [c["validation_score"] for c in record["candidates"]] == [0.4, 0.56]


def test_gepa_full_saves_the_instruction_next_to_the_baseline_and_never_overwrites(
    tmp_path, monkeypatch, capsys
):
    splits_path, prompts_path = tmp_path / "splits.json", tmp_path / "prompts.json"
    splits_path.write_text(json.dumps({"train": [1], "validation": [2], "test": [3]}))
    prompts_path.write_text(json.dumps({"baseline": "frozen"}))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(config, "PROMPTS_PATH", str(prompts_path))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(data, "load_examples", lambda: ([], [TASK]))
    monkeypatch.setattr(
        prompt_optimization, "run_gepa", lambda *a, **k: fake_gepa_result()
    )
    prompt_optimization.gepa_full()
    saved = json.loads(prompts_path.read_text())
    assert saved == {"baseline": "frozen", "gepa": "new"}
    assert json.loads((tmp_path / "gepa.json").read_text())["best_index"] == 1
    assert "GEPA instruction" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        prompt_optimization.gepa_full()


def setup_test_draws(tmp_path, monkeypatch, draws=2):
    tasks = [{**TASK, "key": n, "prompt": f"p{n}"} for n in (1, 2, 3)]
    splits_path, prompts_path = tmp_path / "splits.json", tmp_path / "prompts.json"
    splits_path.write_text(json.dumps({"train": [3], "validation": [], "test": [1, 2]}))
    prompts_path.write_text(json.dumps({"baseline": "base text", "gepa": "gepa text"}))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(config, "PROMPTS_PATH", str(prompts_path))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(config, "TEST_DRAWS", draws)
    monkeypatch.setattr(config, "TEST_WORKERS", 2)
    monkeypatch.setattr(data, "load_examples", lambda: ([], tasks))
    monkeypatch.setattr(prompt_optimization, "make_student", lambda: (None, None))
    monkeypatch.setattr(
        dspy, "Predict", lambda signature: lambda prompt: SimpleNamespace(response=GOOD)
    )


def test_write_baseline_keeps_the_other_saved_instructions_and_records_search_and_sources(
    tmp_path,
):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"gepa": "optimized text"}))
    seen = {}

    def writer(messages, **kwargs):
        seen.update(kwargs)
        text = (
            "The instruction.\n---SOURCES---\nhttps://a.example\n\nhttps://b.example\n"
        )
        return [
            {"text": text, "tool_calls": None}
        ]  # the shape DSPy returns when tools are passed

    prompt_optimization.write_baseline(
        writer, BASELINE_TASKS, BASELINE_SPLITS, path, search=True
    )
    saved = json.loads(path.read_text())
    assert saved["gepa"] == "optimized text"
    assert saved["baseline"] == "The instruction."
    assert saved["writer"]["sources_used"] == ["https://a.example", "https://b.example"]
    assert saved["writer"]["search_requested"] is True
    assert seen["tools"] == [{"googleSearch": {}}]
    assert saved["writer"]["guideline_sources"] == prompts.WRITER_GUIDELINE_SOURCES


def test_write_baseline_refuses_when_a_baseline_is_saved_even_if_other_keys_exist(
    tmp_path,
):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({"baseline": "frozen", "gepa": "g"}))
    with pytest.raises(FileExistsError):
        prompt_optimization.write_baseline(
            fake_writer("new"), BASELINE_TASKS, BASELINE_SPLITS, path
        )
    assert json.loads(path.read_text())["baseline"] == "frozen"


def test_baseline_retries_once_without_search_when_the_search_call_fails(
    tmp_path, monkeypatch, capsys
):
    class FakeGemini:
        def __init__(self):
            self.history = [{"usage": {"total_tokens": 1}, "cost": 0.01}]
            self.calls = []

        def __call__(self, messages, **kwargs):
            self.calls.append(kwargs)
            if kwargs:
                raise dspy.utils.exceptions.LMError("search is not available")
            return ["Plain instruction."]

    gemini = FakeGemini()
    splits_path, prompts_path = tmp_path / "splits.json", tmp_path / "prompts.json"
    splits_path.write_text(json.dumps(BASELINE_SPLITS))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(config, "PROMPTS_PATH", str(prompts_path))
    monkeypatch.setattr(data, "load_examples", lambda: ([], BASELINE_TASKS))
    monkeypatch.setattr(dspy, "LM", lambda *args, **kwargs: gemini)
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    prompt_optimization.baseline()
    saved = json.loads(prompts_path.read_text())
    assert saved["baseline"] == "Plain instruction."
    assert saved["writer"]["search_requested"] is False
    assert len(gemini.calls) == 2
    assert "retrying once without search" in capsys.readouterr().out


SMOKE_LOG = """command: make gepa-smoke
start: 2026-11-10 20:05:23
2026/11/10 20:05:28 INFO dspy.teleprompt.gepa.gepa: Iteration 0: Base program full valset score: 0.33
2026/11/10 20:09:04 INFO dspy.teleprompt.gepa.gepa: Iteration 5: done

candidates 2, validation scores [0.33, 0.83], best 1, metric calls 42
actual: 42 student calls, 5 Gemini calls, Gemini cost $0.0362
exit code: 0
"""


def test_smoke_record_from_log_has_the_shape_of_a_saved_gepa_record():
    candidates = [{"self": "old"}, {"self": "new"}]
    record = prompt_optimization.smoke_record_from_log(SMOKE_LOG, candidates)
    assert [c["validation_score"] for c in record["candidates"]] == [0.33, 0.83]
    assert [c["instruction"] for c in record["candidates"]] == ["old", "new"]
    assert record["best_index"] == 1
    assert record["student"]["calls"] == 42
    assert record["student"]["seconds"] == 216.0  # 20:05:28 to 20:09:04
    assert record["gemini"]["calls"] == 5
    assert record["gemini"]["dollars"] == 0.04
    assert record["gemini"]["prompt_tokens"] is None  # never recorded for this run


def test_smoke_record_writes_the_file_once_and_the_chart_says_tokens_were_not_recorded(
    tmp_path, monkeypatch
):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "smoke.txt").write_text(SMOKE_LOG)
    gepa_dir = tmp_path / "gepa_smoke"
    gepa_dir.mkdir()
    (gepa_dir / "candidates.json").write_text(
        json.dumps([{"self": "old"}, {"self": "new"}])
    )
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    prompt_optimization.smoke_record(
        str(tmp_path / "logs" / "smoke.txt"), str(gepa_dir)
    )
    record = json.loads((tmp_path / "gepa_smoke.json").read_text())
    with pytest.raises(FileExistsError):
        prompt_optimization.smoke_record(
            str(tmp_path / "logs" / "smoke.txt"), str(gepa_dir)
        )
    plots.draw_gepa(record, tmp_path / "gepa_smoke.png")
    assert (tmp_path / "gepa_smoke.png").stat().st_size > 1000
    monkeypatch.chdir(tmp_path)
    plots.main("gepa-smoke")
    assert (tmp_path / "results" / "figures" / "gepa_smoke.png").exists()
