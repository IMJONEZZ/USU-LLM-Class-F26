import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import dspy
import openai
import pytest

from src import config, data, estimate, evaluate, modal_server, optimize, plots, prompts

VENDORED = Path(__file__).parent.parent / "third_party" / "instruction_following_eval"


def test_kv_cache_counts_keys_and_values_for_every_layer_head_and_token():
    # 2 (K and V) * 28 layers * 8 kv heads * 128 head dim * 2 bytes * 1000 tokens
    mib = modal_server.kv_cache_mib(layers=28, kv_heads=8, head_dim=128, tokens=1000)
    assert mib == pytest.approx(109.375)


def test_vllm_command_takes_model_dtype_and_length_from_config():
    command = modal_server.vllm_command()
    assert command[:3] == ["vllm", "serve", config.MODEL_NAME]
    assert command[command.index("--served-model-name") + 1] == config.SERVED_NAME
    assert command[command.index("--dtype") + 1] == config.DTYPE
    assert command[command.index("--max-model-len") + 1] == str(config.MAX_MODEL_LEN)
    assert command[command.index("--generation-config") + 1] == "vllm"


def test_estimate_sums_calls_and_dollars_by_model():
    gemini_row = {
        "model": "gemini",
        "calls": 2,
        "input_tokens": 1000,
        "output_tokens": 500,
        "basis": "assumed",
    }
    student_row = {"model": "student", "calls": 10, "seconds": 100, "basis": "assumed"}
    totals = estimate.estimate([student_row, gemini_row, gemini_row])
    per_call = (
        1000 * config.GEMINI_INPUT_PER_MILLION + 500 * config.GEMINI_OUTPUT_PER_MILLION
    ) / 1e6
    assert totals["student"]["calls"] == 10
    assert totals["student"]["dollars"] == pytest.approx(100 * config.T4_PER_SECOND)
    assert totals["gemini"]["calls"] == 4
    assert totals["gemini"]["dollars"] == pytest.approx(4 * per_call)


TASK = {"key": 1, "prompt": "p", "instruction_id": "punctuation:no_comma", "kwargs": {}}
GOOD, BAD = "no commas here", "one, two"
PARSE_ERROR = dspy.utils.exceptions.AdapterParseError(
    "ChatAdapter", dspy.Signature("prompt -> response"), "raw reply"
)
SERVER_ERROR = dspy.utils.exceptions.LMError("server down")


def scripted(items):
    """A stand-in for `predict`: each call returns, or raises, the next item."""
    items = iter(items)

    def predict(prompt):
        item = next(items)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(response=item)

    return predict


@pytest.mark.parametrize(
    ("item", "outcome"),
    [
        (GOOD, "pass"),
        (BAD, "fail"),
        (PARSE_ERROR, "parse_error"),
        (SERVER_ERROR, "request_failure"),
    ],
)
def test_draw_names_how_each_kind_of_reply_went(item, outcome):
    assert evaluate.draw(scripted([item]), TASK)["outcome"] == outcome


def test_acceptance_rate_is_the_share_of_passes():
    result = evaluate.acceptance_rate(scripted([GOOD] * 5), TASK, 5)
    assert result["pass"] == 5
    assert result["rate"] == 1.0


def test_acceptance_rate_counts_a_missing_marker_as_a_miss():
    result = evaluate.acceptance_rate(scripted([GOOD, BAD, PARSE_ERROR]), TASK, 3)
    assert (result["pass"], result["fail"], result["parse_error"]) == (1, 1, 1)
    assert result["rate"] == pytest.approx(1 / 3)


def test_acceptance_rate_leaves_request_failures_out_of_the_denominator():
    result = evaluate.acceptance_rate(scripted([GOOD, SERVER_ERROR]), TASK, 2)
    assert result["request_failure"] == 1
    assert result["rate"] == 1.0


def test_acceptance_rate_is_none_when_nothing_could_be_graded():
    result = evaluate.acceptance_rate(scripted([SERVER_ERROR] * 2), TASK, 2)
    assert result["rate"] is None


def test_pilot_tasks_never_include_test_tasks():
    tasks = [{"key": 1}, {"key": 2}, {"key": 3}]
    splits = {"train": [1], "validation": [2], "test": [3]}
    chosen = evaluate.pilot_tasks(tasks, splits)
    assert [t["key"] for t in chosen] == [1, 2]
    assert [t["split"] for t in chosen] == ["train", "validation"]


def test_vendored_checker_files_match_recorded_hashes():
    # Google's files must stay byte-for-byte unedited; any change goes in src/checks.py instead
    for line in (VENDORED / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((VENDORED / name).read_bytes()).hexdigest() == expected, (
            name
        )


def test_make_student_points_at_the_server_with_cache_and_json_fallback_off(
    monkeypatch,
):
    monkeypatch.setenv("MODAL_URL", "https://example.test")
    monkeypatch.setenv("MODAL_ID", "id")
    monkeypatch.setenv("MODAL_SECRET", "secret")
    lm, predict = optimize.make_student()
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
    monkeypatch.setattr(optimize, "make_student", lambda: student)
    monkeypatch.setattr(data, "load_examples", lambda: ([], [TASK]))
    splits_path = tmp_path / "splits.json"
    splits_path.write_text('{"train": [1]}')
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    optimize.demo()
    out = capsys.readouterr().out
    assert "reply 1: 'first reply'" in out
    assert "PARSE ERROR, raw reply: raw reply" in out
    assert "the system text" in out
    assert "14 output tokens" in out


def test_wait_until_ready_keeps_polling_health_until_it_answers(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise OSError("not up yet")

    monkeypatch.setattr(modal_server.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(modal_server.time, "sleep", lambda seconds: None)
    modal_server.wait_until_ready()
    assert calls == ["http://localhost:8000/health"] * 3


def test_draw_split_writes_a_chart_file(tmp_path):
    counts = {
        "a:b": {"train": 3, "validation": 2, "test": 2},
        "c:d": {"train": 4, "validation": 3, "test": 3},
    }
    path = tmp_path / "figures" / "split.png"
    plots.draw_split(counts, {"e:f": 5}, path)
    assert path.stat().st_size > 1000


def test_plots_main_draws_from_the_saved_split(tmp_path, monkeypatch):
    tasks = [{**TASK, "key": n} for n in range(1, 8)]
    splits = {"train": [1, 2, 3], "validation": [4, 5], "test": [6, 7]}
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps(splits))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(data, "load_examples", lambda: (tasks, tasks))
    monkeypatch.chdir(tmp_path)
    plots.main()
    assert (tmp_path / "results" / "figures" / "split.png").stat().st_size > 1000


def run_evaluate(mode, monkeypatch, tmp_path, reply=GOOD):
    """Run evaluate.main against a fake model that always answers GOOD, in a scratch folder."""
    tasks = [{**TASK, "key": n, "prompt": f"p{n}"} for n in (1, 2, 3)]
    splits_path = tmp_path / "splits.json"
    splits_path.write_text('{"train": [1, 2], "validation": [3], "test": []}')
    entry = {"outputs": ["[[ ## response ## ]] ok"], "usage": {"completion_tokens": 3}}
    fake_lm = SimpleNamespace(history=[entry] * 6)
    student = (fake_lm, lambda prompt: SimpleNamespace(response=reply))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(config, "PILOT_SAMPLES", 2)
    monkeypatch.setattr(data, "load_examples", lambda: ([], tasks))
    monkeypatch.setattr(optimize, "make_student", lambda: student)
    monkeypatch.chdir(tmp_path)
    evaluate.main(mode)


def test_evaluate_smoke_prints_the_totals_and_the_raw_replies_but_saves_nothing(
    monkeypatch, tmp_path, capsys
):
    run_evaluate("smoke", monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "3 tasks x 2 samples: {'pass': 6, 'fail': 0" in out
    assert "marker ok" in out
    assert not (tmp_path / "results").exists()


def test_evaluate_pilot_saves_each_task_with_its_split_and_never_overwrites(
    monkeypatch, tmp_path
):
    run_evaluate("pilot", monkeypatch, tmp_path)
    saved = json.loads((tmp_path / "results" / "pilot_oneliner.json").read_text())
    assert saved["run"]["totals"]["pass"] == 6
    assert saved["run"]["instruction"] == prompts.ONE_LINER
    assert [t["split"] for t in saved["tasks"]] == ["train", "train", "validation"]
    assert saved["tasks"][0]["rate"] == 1.0
    with pytest.raises(FileExistsError):
        run_evaluate("pilot", monkeypatch, tmp_path)


def test_evaluate_says_stop_when_more_than_the_cutoff_of_tasks_pass_every_sample(
    monkeypatch, tmp_path, capsys
):
    run_evaluate("smoke", monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "100% of tasks passed every sample, 0% passed none" in out
    assert "STOP, reconsider before M5" in out


def test_evaluate_says_continue_when_few_tasks_pass_every_sample(
    monkeypatch, tmp_path, capsys
):
    run_evaluate("smoke", monkeypatch, tmp_path, reply=BAD)
    out = capsys.readouterr().out
    assert "0% of tasks passed every sample, 100% passed none" in out
    assert "continue to M5" in out


PILOT_RECORD = {
    "run": {
        "samples_per_task": 3,
        "totals": {"pass": 5, "fail": 2, "parse_error": 1, "request_failure": 0},
    },
    "tasks": [
        {"rate": 1.0, "split": "train"},
        {"rate": 1.0, "split": "validation"},
        {"rate": 0.67, "split": "train"},
        {"rate": 0.0, "split": "train"},
        {"rate": None, "split": "train"},
    ],
}


def test_draw_pilot_writes_a_chart_file_and_skips_tasks_that_could_not_be_graded(
    tmp_path,
):
    path = tmp_path / "figures" / "pilot.png"
    plots.draw_pilot(PILOT_RECORD, path)
    assert path.stat().st_size > 1000


def test_plots_main_pilot_reads_the_saved_pilot_results(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "pilot_oneliner.json").write_text(json.dumps(PILOT_RECORD))
    monkeypatch.chdir(tmp_path)
    plots.main("pilot")
    assert (tmp_path / "results" / "figures" / "pilot.png").stat().st_size > 1000


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
    optimize.write_baseline(
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
    optimize.write_baseline(fake_writer("one"), BASELINE_TASKS, BASELINE_SPLITS, path)
    with pytest.raises(FileExistsError):
        optimize.write_baseline(
            fake_writer("two"), BASELINE_TASKS, BASELINE_SPLITS, path
        )
    assert json.loads(path.read_text())["baseline"] == "one"


def test_write_baseline_rejects_an_empty_reply_and_saves_nothing(tmp_path):
    path = tmp_path / "prompts.json"
    with pytest.raises(ValueError):
        optimize.write_baseline(
            fake_writer("  "), BASELINE_TASKS, BASELINE_SPLITS, path
        )
    assert not path.exists()


def test_metric_scores_the_parsed_response_and_accepts_gepas_extra_arguments():
    gold = SimpleNamespace(task=TASK)
    good = optimize.metric(gold, SimpleNamespace(response=GOOD), None, "predict", None)
    bad = optimize.metric(gold, SimpleNamespace(response=BAD))
    assert (good.score, bad.score) == (1.0, 0.0)
    assert "punctuation:no_comma" in bad.feedback


@pytest.mark.parametrize(
    "pred",
    [SimpleNamespace(response=""), SimpleNamespace(response=None), SimpleNamespace()],
)
def test_metric_gives_zero_and_feedback_for_an_empty_or_missing_response(pred):
    result = optimize.metric(SimpleNamespace(task=TASK), pred)
    assert result.score == 0.0
    assert "Failed instruction" in result.feedback


def test_gepa_examples_carry_the_task_and_only_the_prompt_as_input():
    tasks = [{**TASK, "key": 1}, {**TASK, "key": 2}]
    examples = optimize.gepa_examples(tasks, [2])
    assert len(examples) == 1
    assert examples[0].task["key"] == 2
    assert list(examples[0].inputs().keys()) == ["prompt"]


@pytest.mark.parametrize(
    "budget", [{}, {"auto": "light", "max_metric_calls": 5}, {"seed": 1}]
)
def test_build_gepa_needs_exactly_one_budget_setting(budget):
    with pytest.raises(ValueError):
        optimize.build_gepa(None, "logs", **budget)


def test_build_gepa_passes_the_decided_settings_through(monkeypatch):
    seen = {}
    monkeypatch.setattr(dspy, "GEPA", lambda **kwargs: seen.update(kwargs))
    optimize.build_gepa("reflection", "logs", auto="light")
    assert seen["auto"] == "light"
    assert seen["add_format_failure_as_feedback"] is True
    assert seen["track_stats"] is True
    assert seen["seed"] == config.SEED
    assert seen["log_dir"] == "logs"
    assert seen["reflection_lm"] == "reflection"
    assert seen["metric"] is optimize.metric


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
        optimize, "run_gepa", lambda *a, **k: (optimized, student, reflection)
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"train": list(range(1, 13))}))
    monkeypatch.setattr(config, "SPLITS_PATH", str(splits_path))
    monkeypatch.setattr(data, "load_examples", lambda: ([], [TASK]))
    optimize.gepa_smoke()
    out = capsys.readouterr().out
    assert "candidate 0: a new instruction" in out
    assert "the reflection prompt" in out
    assert "3 student calls, 1 Gemini calls" in out


def test_gepa_light_budget_matches_the_dspy_formula():
    assert estimate.gepa_light_calls(90) == 740  # 4 * 90 + 380


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
    record = optimize.summarize_gepa(
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
    monkeypatch.setattr(optimize, "run_gepa", lambda *a, **k: fake_gepa_result())
    optimize.gepa_full()
    saved = json.loads(prompts_path.read_text())
    assert saved == {"baseline": "frozen", "gepa": "new"}
    assert json.loads((tmp_path / "gepa.json").read_text())["best_index"] == 1
    assert "GEPA instruction" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        optimize.gepa_full()


def test_draw_keeps_the_reply_text_for_every_outcome():
    assert evaluate.draw(scripted([GOOD]), TASK)["reply"] == GOOD
    assert evaluate.draw(scripted([PARSE_ERROR]), TASK)["reply"] == "raw reply"
    assert evaluate.draw(scripted([SERVER_ERROR]), TASK)["reply"] == "server down"


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
    monkeypatch.setattr(optimize, "make_student", lambda: (None, None))
    monkeypatch.setattr(
        dspy, "Predict", lambda signature: lambda prompt: SimpleNamespace(response=GOOD)
    )


def test_run_test_draws_saves_every_draw_for_each_instruction_and_test_task(
    tmp_path, monkeypatch, capsys
):
    setup_test_draws(tmp_path, monkeypatch)
    evaluate.run_test_draws()
    lines = (tmp_path / "results" / "test_draws.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert sorted((r["prompt"], r["key"]) for r in rows) == [
        (name, key) for name in ("baseline", "gepa", "one_liner") for key in (1, 2)
    ]  # task 3 is a train task and is never drawn
    assert all(len(r["draws"]) == 2 for r in rows)
    assert rows[0]["draws"][0] == {"outcome": "pass", "reply": GOOD}
    assert "finished 6 of 6" in capsys.readouterr().out


def test_run_test_draws_resumes_without_redoing_finished_work_and_rejects_changed_settings(
    tmp_path, monkeypatch
):
    setup_test_draws(tmp_path, monkeypatch)
    evaluate.run_test_draws()
    monkeypatch.setattr(
        dspy,
        "Predict",
        lambda signature: lambda prompt: 1 / 0,  # would crash if any cell were redone
    )
    evaluate.run_test_draws()
    lines = (tmp_path / "results" / "test_draws.jsonl").read_text().splitlines()
    assert len(lines) == 6
    monkeypatch.setattr(config, "TEST_DRAWS", 3)
    with pytest.raises(ValueError):
        evaluate.run_test_draws()


GEPA_RECORD = {
    "best_index": 1,
    "candidates": [
        {"index": 0, "validation_score": 0.4, "instruction": "old"},
        {"index": 1, "validation_score": 0.56, "instruction": "new"},
    ],
    "student": {"calls": 700, "seconds": 1100.0, "dollars": 0.18},
    "gemini": {
        "calls": 15,
        "prompt_tokens": 20000,
        "completion_tokens": 30000,
        "dollars": 0.12,
    },
}


def cell(prompt, kind, outcomes):
    return {
        "prompt": prompt,
        "key": 1,
        "instruction_id": kind,
        "draws": [{"outcome": o, "reply": ""} for o in outcomes],
    }


DRAW_ROWS = [
    cell(name, kind, ["pass", "fail", "request_failure"])
    for name in ("one_liner", "baseline", "gepa")
    for kind in ("a:type", "b:type")
]


def test_draw_gepa_writes_a_chart_file(tmp_path):
    path = tmp_path / "gepa.png"
    plots.draw_gepa(GEPA_RECORD, path)
    assert path.stat().st_size > 1000


def test_draw_draws_writes_a_chart_file_and_ignores_request_failures(tmp_path):
    path = tmp_path / "draws.png"
    plots.draw_draws(DRAW_ROWS, path)
    assert path.stat().st_size > 1000


def test_plots_main_gepa_and_draws_read_the_saved_results(tmp_path, monkeypatch):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "gepa.json").write_text(json.dumps(GEPA_RECORD))
    lines = [json.dumps(r) for r in DRAW_ROWS]
    (tmp_path / "results" / "test_draws.jsonl").write_text("\n".join(lines))
    monkeypatch.chdir(tmp_path)
    plots.main("gepa")
    plots.main("draws")
    assert (tmp_path / "results" / "figures" / "gepa.png").exists()
    assert (tmp_path / "results" / "figures" / "draws.png").exists()


MARKED = "[[ ## response ## ]]\nno commas here\n\n[[ ## completed ## ]]"


def fake_client(content=MARKED, logprobs=(-1.0, -3.0), error=None):
    seen = {}

    def create(**kwargs):
        seen.update(kwargs)
        if error:
            raise error
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content),
            logprobs=SimpleNamespace(
                content=[SimpleNamespace(logprob=x) for x in logprobs]
            ),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return client, seen


def test_average_logprob_covers_the_generated_tokens_and_survives_zero_tokens():
    assert evaluate.average_logprob([-1.0, -3.0]) == -2.0
    assert evaluate.average_logprob([]) is None


def test_greedy_asks_at_temperature_zero_with_the_messages_dspy_builds():
    client, seen = fake_client()
    result = evaluate.greedy(client, "an instruction", TASK)
    built = dspy.ChatAdapter(use_json_adapter_fallback=False).format(
        dspy.Signature(prompts.SIGNATURE, "an instruction"),
        [],
        {"prompt": TASK["prompt"]},
    )
    assert seen["messages"] == built
    assert seen["temperature"] == 0
    assert seen["logprobs"] is True
    assert result == {
        "outcome": "pass",
        "reply": "no commas here",
        "avg_logprob": -2.0,
        "tokens": 2,
        "finish": "stop",
    }


def test_greedy_names_a_missing_marker_a_parse_error_and_keeps_the_raw_reply():
    client, _ = fake_client(content="no marker, sorry")
    result = evaluate.greedy(client, "x", TASK)
    assert result["outcome"] == "parse_error"
    assert result["reply"] == "no marker, sorry"


def test_greedy_reports_a_server_failure_and_a_reply_with_no_tokens():
    broken, _ = fake_client(error=openai.OpenAIError("down"))
    assert evaluate.greedy(broken, "x", TASK)["outcome"] == "request_failure"
    silent, _ = fake_client(logprobs=())
    result = evaluate.greedy(silent, "x", TASK)
    assert (result["avg_logprob"], result["tokens"]) == (None, 0)


def test_run_greedy_saves_every_answer_and_checks_reruns_and_the_dspy_messages(
    tmp_path, monkeypatch, capsys
):
    setup_test_draws(tmp_path, monkeypatch)
    for name, value in (
        ("MODAL_URL", "https://example.test"),
        ("MODAL_ID", "i"),
        ("MODAL_SECRET", "s"),
    ):
        monkeypatch.setenv(name, value)
    client, _ = fake_client()
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: client)
    built = dspy.ChatAdapter(use_json_adapter_fallback=False).format(
        dspy.Signature(prompts.SIGNATURE, prompts.ONE_LINER), [], {"prompt": "p1"}
    )
    fake_lm = SimpleNamespace(history=[{"messages": built}])
    monkeypatch.setattr(
        optimize, "make_student", lambda instruction: (fake_lm, lambda prompt: None)
    )
    evaluate.run_greedy()
    saved = json.loads((tmp_path / "results" / "greedy.json").read_text())
    assert len(saved["results"]) == 6  # 3 instructions x 2 test tasks
    assert saved["run"]["rerun_different_replies"] == 0
    assert saved["run"]["messages_match_dspy"] is True
    assert saved["results"][0]["avg_logprob"] == -2.0
    assert "messages match DSPy: True" in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        evaluate.run_greedy()


def draw_row(prompt, key, kind, outcomes):
    return {
        "prompt": prompt,
        "key": key,
        "instruction_id": kind,
        "draws": [{"outcome": o, "reply": ""} for o in outcomes],
    }


def test_cell_rates_leave_out_request_failures_and_keep_integer_task_keys():
    rows = [draw_row("gepa", 1, "a:t", ["pass", "fail", "request_failure"])]
    from_disk = [json.loads(json.dumps(r)) for r in rows]
    assert evaluate.cell_rates(from_disk) == {"gepa": {1: 0.5}}


def test_paired_differences_pair_by_task_key_and_skip_tasks_missing_from_one_instruction():
    rates = {
        "baseline": {3: 0.25, 1: 0.5, 2: 1.0},
        "gepa": {2: 0.5, 1: 1.0},
    }  # other order, task 3 missing
    keys, diffs = evaluate.paired_differences(rates, "baseline", "gepa")
    assert keys == [1, 2]
    assert diffs == [0.5, -0.5]


def test_bootstrap_interval_repeats_with_a_seed_and_collapses_when_all_differences_are_equal():
    diffs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]
    assert evaluate.bootstrap_interval(diffs, 2000, 7) == evaluate.bootstrap_interval(
        diffs, 2000, 7
    )
    assert evaluate.bootstrap_interval(diffs, 2000, 7) != evaluate.bootstrap_interval(
        diffs, 2000, 8
    )
    assert evaluate.bootstrap_interval([0.5] * 10, 500, 1) == (0.5, 0.5)


def test_the_by_hand_bootstrap_agrees_with_scipy_within_sampling_noise():
    diffs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.25, 0.4, -0.05, 0.15] * 3
    hand = evaluate.bootstrap_interval(diffs, 5000, 3)
    library = evaluate.scipy_interval([0.0] * 30, diffs, 5000, 3)
    assert hand == pytest.approx(library, abs=0.05)


def test_screenshot_task_takes_the_largest_gain_and_breaks_ties_toward_the_lower_one_liner_rate():
    rates = {
        "one_liner": {1: 0.5, 2: 0.0, 3: 0.25},
        "gepa": {1: 1.0, 2: 0.5, 3: 0.75},
    }  # every gain is 0.5
    assert evaluate.screenshot_task(rates) == 2
    rates["gepa"][3] = 1.0  # task 3 now gains 0.75
    assert evaluate.screenshot_task(rates) == 3


def test_analyze_writes_the_summary_with_small_type_groups_and_never_overwrites(
    tmp_path, monkeypatch, capsys
):
    kinds = {
        1: "a:t",
        2: "a:t",
        3: "b:t",
        4: "b:t",
        5: "b:t",
    }  # one group of 2 tasks and one of 3
    passes = {
        "one_liner": lambda k: max(k - 2, 0),
        "baseline": lambda k: max(k - 1, 0),
        "gepa": lambda k: k,
    }
    rows = [
        draw_row(
            name,
            key,
            kinds[key],
            ["pass"] * passes[name](key) + ["fail"] * (5 - passes[name](key)),
        )
        for name in passes
        for key in kinds
    ]
    greedy_rows = [
        {
            "prompt": name,
            "key": key,
            "instruction_id": kinds[key],
            "outcome": "pass",
            "reply": f"{name} reply",
            "avg_logprob": -0.1 * key - 0.02 * index,
        }
        for index, name in enumerate(passes)
        for key in kinds
    ]
    results = tmp_path / "results"
    results.mkdir()
    (results / "test_draws.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (results / "greedy.json").write_text(json.dumps({"results": greedy_rows}))
    tasks = [
        {**TASK, "key": key, "prompt": f"prompt {key}", "instruction_id": kinds[key]}
        for key in kinds
    ]
    monkeypatch.setattr(config, "RESULTS_DIR", str(results))
    monkeypatch.setattr(config, "BOOTSTRAP_RESAMPLES", 200)
    monkeypatch.setattr(data, "load_examples", lambda: ([], tasks))
    evaluate.analyze()
    summary = json.loads((results / "summary.json").read_text())
    assert {t: v["tasks"] for t, v in summary["per_type"].items()} == {
        "a:t": 2,
        "b:t": 3,
    }
    assert [c["from"] for c in summary["comparisons"]] == ["baseline", "one_liner"]
    assert (
        summary["comparisons"][0]["mean_difference"] > 0
    )  # GEPA passes more than the baseline in this fake data
    assert (
        summary["screenshot_task"]["key"] == 2
    )  # gains tie at 0.4 for tasks 2 to 5; task 2 has the lowest one-liner rate
    assert summary["spearman_logprob_vs_rate"]["all"]["points"] == 15
    out = capsys.readouterr().out
    assert "[gepa] greedy pass:" in out
    with pytest.raises(FileExistsError):
        evaluate.analyze()


def test_screenshot_task_treats_gains_that_differ_only_by_floating_point_noise_as_ties():
    # 0.7 - 0.3 is 0.39999999999999997 while 0.9 - 0.5 is 0.4, but both gains are really 0.4
    rates = {"one_liner": {1: 0.3, 2: 0.5}, "gepa": {1: 0.7, 2: 0.9}}
    assert evaluate.screenshot_task(rates) == 1  # the lower one-liner rate wins the tie


SUMMARY = {
    "comparisons": [
        {
            "from": "baseline",
            "to": "gepa",
            "tasks": 75,
            "mean_difference": 0.08,
            "by_hand_interval": [0.02, 0.14],
            "scipy_interval": [0.03, 0.14],
        },
        {
            "from": "one_liner",
            "to": "gepa",
            "tasks": 75,
            "mean_difference": 0.15,
            "by_hand_interval": [0.08, 0.22],
            "scipy_interval": [0.09, 0.22],
        },
    ],
    "points": [
        {"prompt": name, "key": key, "avg_logprob": -0.1 * key, "rate": 0.1 * key}
        for name in ("one_liner", "baseline", "gepa")
        for key in (1, 2, 3)
    ],
    "spearman_logprob_vs_rate": {"all": {"rho": -0.5, "points": 9}},
}


def test_draw_summary_writes_a_chart_file_and_main_reads_the_saved_summary(
    tmp_path, monkeypatch
):
    plots.draw_summary(SUMMARY, tmp_path / "summary.png")
    assert (tmp_path / "summary.png").stat().st_size > 1000
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "summary.json").write_text(json.dumps(SUMMARY))
    monkeypatch.chdir(tmp_path)
    plots.main("summary")
    assert (tmp_path / "results" / "figures" / "summary.png").exists()


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

    optimize.write_baseline(writer, BASELINE_TASKS, BASELINE_SPLITS, path, search=True)
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
        optimize.write_baseline(
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
    optimize.baseline()
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
    record = optimize.smoke_record_from_log(SMOKE_LOG, candidates)
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
    optimize.smoke_record(str(tmp_path / "logs" / "smoke.txt"), str(gepa_dir))
    record = json.loads((tmp_path / "gepa_smoke.json").read_text())
    with pytest.raises(FileExistsError):
        optimize.smoke_record(str(tmp_path / "logs" / "smoke.txt"), str(gepa_dir))
    plots.draw_gepa(record, tmp_path / "gepa_smoke.png")
    assert (tmp_path / "gepa_smoke.png").stat().st_size > 1000
    monkeypatch.chdir(tmp_path)
    plots.main("gepa-smoke")
    assert (tmp_path / "results" / "figures" / "gepa_smoke.png").exists()
