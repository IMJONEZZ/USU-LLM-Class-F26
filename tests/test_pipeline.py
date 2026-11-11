import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import dspy
import openai
import pytest

from src import (
    config,
    data,
    estimate,
    evaluate,
    modal_server,
    plots,
    prompt_optimization,
    prompts,
)

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
    monkeypatch.setattr(prompt_optimization, "make_student", lambda: student)
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


def test_gepa_light_budget_matches_the_dspy_formula():
    assert estimate.gepa_light_calls(90) == 740  # 4 * 90 + 380


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
    monkeypatch.setattr(prompt_optimization, "make_student", lambda: (None, None))
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
        prompt_optimization,
        "make_student",
        lambda instruction: (fake_lm, lambda prompt: None),
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


def test_run_test_draws_lets_the_worker_count_change_on_a_resume_and_records_the_change(
    tmp_path, monkeypatch
):
    setup_test_draws(tmp_path, monkeypatch)
    evaluate.run_test_draws()  # runs with 2 workers
    monkeypatch.setattr(config, "TEST_WORKERS", 5)
    evaluate.run_test_draws()
    meta = json.loads((tmp_path / "results" / "test_draws_meta.json").read_text())
    assert meta["workers"] == 5
    assert meta["earlier_workers"] == [2]
    monkeypatch.setattr(config, "TEST_WORKERS", 5)
    evaluate.run_test_draws()  # same count again keeps the history
    meta = json.loads((tmp_path / "results" / "test_draws_meta.json").read_text())
    assert meta["earlier_workers"] == [2]


def test_draw_greedy_writes_a_chart_file_and_skips_answers_without_a_confidence(
    tmp_path, monkeypatch
):
    rows = [
        {
            "prompt": name,
            "key": key,
            "outcome": outcome,
            "avg_logprob": -0.1 * key if key else None,
        }
        for name in ("one_liner", "baseline", "gepa")
        for key, outcome in enumerate(["pass", "fail", "parse_error", "pass"])
    ]
    plots.draw_greedy(rows, tmp_path / "greedy.png")
    assert (tmp_path / "greedy.png").stat().st_size > 1000
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "greedy.json").write_text(json.dumps({"results": rows}))
    monkeypatch.chdir(tmp_path)
    plots.main("greedy")
    assert (tmp_path / "results" / "figures" / "greedy.png").exists()
