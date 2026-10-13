import importlib
import socket

import httpx
import numpy as np
import pytest
from google import genai
from google.genai import errors, types

from src import config, data, teacher
from src import eval as evaluation
from tests.helpers import (
    LONG_Q,
    QUESTIONS,
    ROW,
    _build,
    _cands,
    _eval_paths,
    _fake_generate_chain,
    _grade_row,
    _paths,
    _qrow,
    _source_row,
    _Spend,
    _teacher_chain,
    _train_rows,
    _trow,
    _without_line,
)


# ---- import safety ----
def test_import_config_and_teacher_no_network_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("network touched at import")

    monkeypatch.setattr(socket.socket, "connect", boom)
    from src import teacher

    importlib.reload(config)
    importlib.reload(teacher)


def test_teacher_call_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from src import teacher

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        teacher.get_client()


# ---- M3: teacher (every API call is faked) ----
_PER_DAY_QUOTA = [
    {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
        ],
    }
]


def _api_error(cls, code, status, details=None):
    body = {"error": {"message": "x", "status": status}}
    if details:
        body["error"]["details"] = details
    return cls(code, body)


def _response(text="ok", finish=types.FinishReason.STOP):
    part = types.Part(text=text)
    candidate = types.Candidate(
        content=types.Content(parts=[part]), finish_reason=finish
    )
    return types.GenerateContentResponse(candidates=[candidate])


class _FakeClient:
    def __init__(self, outcome):
        self.calls = []
        self.models = self
        self._outcome = outcome

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _use_client(monkeypatch, outcome):
    client = _FakeClient(outcome)
    monkeypatch.setattr(teacher, "get_client", lambda: client)
    return client


def test_get_client_with_key_builds_a_client_without_network(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def boom(*a, **k):
        raise AssertionError("network touched while building the client")

    monkeypatch.setattr(socket.socket, "connect", boom)
    assert isinstance(teacher.get_client(), genai.Client)


# ---- _call_model: SDK responses and errors become text or TeacherAPIError ----
def test_call_model_returns_text_and_passes_settings(monkeypatch):
    client = _use_client(monkeypatch, _response("hello"))
    assert teacher._call_model("m-1", "the prompt", 0.5) == "hello"
    call = client.calls[0]
    assert call["model"] == "m-1"
    assert call["contents"] == "the prompt"
    assert call["config"].temperature == 0.5
    assert call["config"].max_output_tokens == config.TEACHER_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_api_error(errors.ClientError, 401, "UNAUTHENTICATED"), "auth"),
        (_api_error(errors.ClientError, 403, "PERMISSION_DENIED"), "auth"),
        (_api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "rate_limit"),
        (
            _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", _PER_DAY_QUOTA),
            "quota",
        ),
        (_api_error(errors.ClientError, 400, "INVALID_ARGUMENT"), "bad_request"),
        (_api_error(errors.ServerError, 503, "UNAVAILABLE"), "server"),
        (httpx.ReadTimeout("slow"), "timeout"),
    ],
)
def test_call_model_maps_errors_to_kinds(monkeypatch, error, kind):
    _use_client(monkeypatch, error)
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == kind


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (_response(""), "empty"),
        (types.GenerateContentResponse(candidates=[]), "empty"),
        (_response("cut off", types.FinishReason.MAX_TOKENS), "truncated"),
        (_response("nope", types.FinishReason.SAFETY), "blocked"),
        (
            types.GenerateContentResponse(
                prompt_feedback=types.GenerateContentResponsePromptFeedback(
                    block_reason=types.BlockedReason.SAFETY
                )
            ),
            "blocked",
        ),
    ],
)
def test_call_model_rejects_unusable_responses(monkeypatch, response, kind):
    _use_client(monkeypatch, response)
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == kind


def test_call_model_missing_key_message_is_clear(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        teacher._call_model("m", "p", 0.0)


# ---- _generate: retry only the transient failures ----
def _script_call_model(monkeypatch, outcomes):
    calls, sleeps = [], []
    queue = list(outcomes)

    def fake(model, prompt, temperature):
        calls.append(prompt)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_call_model", fake)
    monkeypatch.setattr(teacher.time, "sleep", sleeps.append)
    return calls, sleeps


def test_generate_retries_transient_errors_then_succeeds(monkeypatch):
    calls, sleeps = _script_call_model(
        monkeypatch,
        [
            teacher.TeacherAPIError("rate_limit"),
            teacher.TeacherAPIError("timeout"),
            "fine",
        ],
    )
    assert teacher._generate("p", "m", 0.0) == "fine"
    assert len(calls) == 3
    assert sleeps == [config.TEACHER_RETRY_SECONDS] * 2


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "server"])
def test_generate_gives_up_after_max_retries(monkeypatch, kind):
    attempts = 1 + config.TEACHER_MAX_RETRIES
    calls, _ = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError(kind)] * attempts
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == attempts


@pytest.mark.parametrize(
    "kind", ["auth", "quota", "bad_request", "empty", "blocked", "truncated"]
)
def test_generate_does_not_retry_permanent_errors(monkeypatch, kind):
    calls, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError(kind), "never reached"]
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == 1
    assert sleeps == []


# ---- chain_prompt ----
def test_chain_prompt_has_question_key_and_all_section_labels():
    prompt = teacher.chain_prompt("How many sheep are left?", "zebra-77")
    assert "How many sheep are left?" in prompt
    assert "zebra-77" in prompt
    assert all(label in prompt for label in config.SECTION_LABELS)


def test_chain_prompt_short_key_must_be_the_final_answer_verbatim():
    prompt = teacher.chain_prompt("Q?", "zebra-77")
    assert f"{config.FINAL_ANSWER_LABEL} zebra-77" in prompt


def test_chain_prompt_long_key_asks_for_a_short_final_answer():
    key = "A long rebuttal sentence. " * 5
    assert len(key) > config.LONG_KEY_CHARS
    prompt = teacher.chain_prompt("Q?", key)
    assert key in prompt
    assert f"{config.FINAL_ANSWER_LABEL} {key}" not in prompt
    assert "short" in prompt


def _script_generate(monkeypatch, outcomes):
    prompts, models = [], []
    queue = list(outcomes)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        models.append(model)
        outcome = queue.pop(0) if queue else teacher.TeacherAPIError("empty")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, models


def test_generate_chain_adds_only_a_chain_to_the_row(monkeypatch):
    chain = _teacher_chain("9")
    prompts, models = _script_generate(monkeypatch, [chain])
    assert teacher.generate_chain(ROW) == {**ROW, "chain": chain}
    assert len(prompts) == 1
    assert models == [config.TEACHER_MODEL]


def test_generate_chain_prompt_has_question_and_the_correct_answer(monkeypatch):
    row = {**ROW, "answer": "zebra-77"}
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("zebra-77")])
    teacher.generate_chain(row)
    assert "How many sheep are left?" in prompts[0]
    assert "zebra-77" in prompts[0]


def test_generate_chain_retries_until_a_chain_is_valid(monkeypatch):
    good = _teacher_chain("9")
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("12"), good])
    assert teacher.generate_chain(ROW)["chain"] == good
    assert len(prompts) == 2


def test_generate_chain_returns_none_after_all_attempts_fail(monkeypatch):
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("12")] * 20)
    assert teacher.generate_chain(ROW) is None
    assert len(prompts) == config.TEACHER_ATTEMPTS


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated"])
def test_generate_chain_skips_content_level_failures(monkeypatch, kind):
    good = _teacher_chain("9")
    _script_generate(monkeypatch, [teacher.TeacherAPIError(kind), good])
    assert teacher.generate_chain(ROW)["chain"] == good


@pytest.mark.parametrize(
    "kind", ["auth", "quota", "rate_limit", "timeout", "server", "bad_request"]
)
def test_generate_chain_stops_on_errors_that_survive_the_retries(monkeypatch, kind):
    prompts, _ = _script_generate(
        monkeypatch, [teacher.TeacherAPIError(kind, "why"), _teacher_chain("9")]
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher.generate_chain(ROW)
    assert caught.value.kind == kind
    assert len(prompts) == 1


def test_generate_chain_accepts_alternative_answers(monkeypatch):
    _script_generate(monkeypatch, [_teacher_chain("nine")])
    assert teacher.generate_chain(ROW) is not None


@pytest.mark.parametrize(
    "wrapper", ["```\n{c}\n```", "```text\n{c}\n```", "  \n{c}\n\n"]
)
def test_generate_chain_unwraps_code_fences_and_whitespace(monkeypatch, wrapper):
    chain = _teacher_chain("9")
    _script_generate(monkeypatch, [wrapper.format(c=chain)])
    assert teacher.generate_chain(ROW)["chain"] == chain


def _swap_lines(chain, i, j):
    lines = chain.split("\n")
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


@pytest.mark.parametrize(
    "bad",
    [
        "The answer is 9.",
        _without_line(_teacher_chain("9"), "Final answer:") + "\nAnswer: 9",
        _swap_lines(_teacher_chain("9"), 1, 2),
        "Restate: a\nCheck: b\nSolve: c\nVerify: d\nFinal answer: 9",
        _teacher_chain("9", "x" * 2000),
        _teacher_chain("9", "### Response:"),
    ],
    ids=[
        "no-labels",
        "no-final-label",
        "wrong-order",
        "too-short",
        "too-long",
        "marker",
    ],
)
def test_generate_chain_rejects_malformed_chains(monkeypatch, bad):
    _script_generate(monkeypatch, [bad] * 20)
    assert teacher.generate_chain(ROW) is None


@pytest.mark.parametrize(
    ("key", "final", "accepted"),
    [
        ("this is impossible", "This is impossible.", True),
        ("5 km", "5 km", True),
        ("5 km", "5", False),
        ("5", "5 km", False),
        ("yes", "Yes, because it is", False),
    ],
)
def test_generate_chain_short_keys_must_match_the_final_answer(
    monkeypatch, key, final, accepted
):
    _script_generate(monkeypatch, [_teacher_chain(final)] * 20)
    row = {**ROW, "answer": key, "accepted_alternatives": []}
    assert (teacher.generate_chain(row) is not None) is accepted


def test_generate_chain_long_keys_accept_any_valid_chain(monkeypatch):
    key = "A long rebuttal that no model would repeat word for word. " * 2
    assert len(key) > config.LONG_KEY_CHARS
    chain = _teacher_chain("A short version")
    _script_generate(monkeypatch, [chain])
    row = {**ROW, "answer": key, "accepted_alternatives": []}
    assert teacher.generate_chain(row)["chain"] == chain


def test_generate_chain_does_not_mutate_the_row_and_result_is_valid(monkeypatch):
    before = dict(ROW)
    _script_generate(monkeypatch, [_teacher_chain("9")])
    result = teacher.generate_chain(ROW)
    assert ROW == before
    data.check_schema(result)


def _ids(path):
    return [r["id"] for r in data.read_jsonl(path)]


def test_build_training_data_writes_rows_in_input_order_and_fills_the_store(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]
    assert _ids(store) == ["r-0", "r-1", "r-2", "r-3"]
    assert all("chain" in r for r in data.read_jsonl(store))
    assert (summary["written"], summary["generated"], summary["reused"]) == (4, 4, 0)


def test_a_second_run_makes_no_calls_and_gives_the_same_file(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    first = out.read_text(encoding="utf-8")
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert seen == []
    assert out.read_text(encoding="utf-8") == first
    assert (summary["generated"], summary["reused"]) == (0, 4)
    assert summary["timing"] == {}


def test_extending_with_more_rows_generates_only_the_new_ones(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(6), out, store)
    assert [r["id"] for r in seen] == ["r-4", "r-5"]
    assert _ids(out) == [f"r-{i}" for i in range(6)]
    assert (summary["generated"], summary["reused"]) == (2, 4)


def test_a_crash_keeps_finished_rows_and_the_rerun_only_does_the_rest(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, crash_on="r-2")
    with pytest.raises(RuntimeError, match="boom"):
        teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(store) == ["r-0", "r-1"]
    assert not out.exists()
    seen = _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    assert [r["id"] for r in seen] == ["r-2", "r-3"]
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit"])
def test_build_training_data_api_error_stops_and_keeps_earlier_rows(
    tmp_path, monkeypatch, kind
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(
        monkeypatch, crash_on="r-2", crash_with=teacher.TeacherAPIError(kind, "why")
    )
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(store) == ["r-0", "r-1"]
    assert not out.exists()


def test_a_row_that_failed_is_recorded_and_never_retried(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    first = teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(out) == ["r-0", "r-2", "r-3"]
    failure = next(r for r in data.read_jsonl(store) if r["id"] == "r-1")
    assert failure["failed"] is True
    assert "chain" not in failure
    assert first["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]

    seen = _fake_generate_chain(monkeypatch)
    second = teacher.build_training_data(_train_rows(4), out, store)
    assert seen == []
    assert second["generated"] == 0
    assert second["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]
    assert _ids(out) == ["r-0", "r-2", "r-3"]


@pytest.mark.parametrize("field", ["question", "answer"])
def test_a_changed_row_for_a_stored_id_stops_before_any_call(
    tmp_path, monkeypatch, field
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    before = store.read_text(encoding="utf-8")
    changed = _train_rows(3)
    changed[0] = {**changed[0], field: "Something else entirely?"}
    seen = _fake_generate_chain(monkeypatch)
    with pytest.raises(ValueError, match="r-0"):
        teacher.build_training_data(changed, out, store)
    assert seen == []
    assert store.read_text(encoding="utf-8") == before


def test_rows_in_the_store_that_are_not_requested_are_kept_but_not_written(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    assert seen == []
    assert _ids(out) == ["r-0", "r-1"]
    assert _ids(store) == ["r-0", "r-1", "r-2", "r-3"]


def test_the_train_file_is_rewritten_freely_because_it_costs_nothing(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    assert _ids(out) == ["r-0", "r-1"]


def test_build_training_data_empty_input_raises_and_writes_nothing(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        teacher.build_training_data([], out, store)
    assert not out.exists()
    assert not store.exists()


def test_build_training_data_duplicate_ids_raise(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    rows = _train_rows(2)
    rows[1] = {**rows[1], "id": rows[0]["id"]}
    with pytest.raises(ValueError, match="id"):
        teacher.build_training_data(rows, out, store)


def test_build_training_data_progress_is_called_once_per_input_row(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    ticks = []
    teacher.build_training_data(
        _train_rows(4), out, store, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 4
    teacher.build_training_data(
        _train_rows(4), out, store, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 8


# ---- timing of API calls ----
def _fake_clock(monkeypatch, step):
    state = {"t": -step}

    def now():
        state["t"] += step
        return state["t"]

    monkeypatch.setattr(teacher, "_now", now)


def test_generate_logs_one_record_per_attempt_with_outcome_and_seconds(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, [teacher.TeacherAPIError("rate_limit"), "fine"])
    _fake_clock(monkeypatch, 2.5)
    assert teacher._generate("p", "m-1", 0.0) == "fine"
    assert teacher.call_log() == [
        {"model": "m-1", "seconds": 2.5, "outcome": "rate_limit"},
        {"model": "m-1", "seconds": 2.5, "outcome": "ok"},
    ]


def test_generate_logs_unexpected_exceptions_and_still_raises(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, [RuntimeError("weird")])
    _fake_clock(monkeypatch, 1.0)
    with pytest.raises(RuntimeError, match="weird"):
        teacher._generate("p", "m-1", 0.0)
    assert teacher.call_log() == [
        {"model": "m-1", "seconds": 1.0, "outcome": "RuntimeError"}
    ]


def test_call_log_returns_a_copy_and_reset_clears_it(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, ["fine"])
    teacher._generate("p", "m", 0.0)
    teacher.call_log().clear()
    assert len(teacher.call_log()) == 1
    teacher.reset_call_log()
    assert teacher.call_log() == []


def test_timing_summary_groups_by_model():
    records = [
        {"model": "pro", "seconds": 10.0, "outcome": "ok"},
        {"model": "pro", "seconds": 30.0, "outcome": "timeout"},
        {"model": "flash", "seconds": 1.0, "outcome": "ok"},
    ]
    assert teacher.timing_summary(records) == {
        "pro": {
            "calls": 2,
            "total_seconds": 40.0,
            "mean_seconds": 20.0,
            "max_seconds": 30.0,
            "outcomes": {"ok": 1, "timeout": 1},
        },
        "flash": {
            "calls": 1,
            "total_seconds": 1.0,
            "mean_seconds": 1.0,
            "max_seconds": 1.0,
            "outcomes": {"ok": 1},
        },
    }


def test_timing_summary_of_nothing_is_empty():
    assert teacher.timing_summary([]) == {}


# ---- M5: paraphrase_question (the teacher writes the rewordings) ----
def _paraphrase_with(monkeypatch, *outputs):
    prompts, settings = [], []
    queue = list(outputs)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        settings.append((model, temperature))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, settings


def test_paraphrase_question_returns_the_rewordings_from_json(monkeypatch):
    prompts, settings = _paraphrase_with(
        monkeypatch, '{"rewordings": ["First way to ask?", "Second way to ask?"]}'
    )
    out = teacher.paraphrase_question("How do planes stay in the air?", 2)
    assert out == ["First way to ask?", "Second way to ask?"]
    assert "How do planes stay in the air?" in prompts[0]
    assert "2" in prompts[0]
    assert settings == [(config.TEACHER_MODEL, config.PARAPHRASE_TEMPERATURE)]


def test_paraphrase_prompt_asks_to_keep_numbers_units_and_negations(monkeypatch):
    prompts, _ = _paraphrase_with(monkeypatch, '{"rewordings": []}')
    teacher.paraphrase_question("Q?", 3)
    text = prompts[0].lower()
    assert "number" in text and "unit" in text and "negation" in text


def test_paraphrase_question_accepts_json_in_code_fences(monkeypatch):
    _paraphrase_with(monkeypatch, '```json\n{"rewordings": ["One way to ask?"]}\n```')
    assert teacher.paraphrase_question("Q?", 1) == ["One way to ask?"]


def test_paraphrase_question_keeps_only_non_blank_strings(monkeypatch):
    _paraphrase_with(monkeypatch, '{"rewordings": ["Good one?", 5, "", "  ", null]}')
    assert teacher.paraphrase_question("Q?", 3) == ["Good one?"]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "",
        "[]",
        '{"other": ["x"]}',
        '{"rewordings": "one string"}',
        '{"rewordings": null}',
    ],
)
def test_paraphrase_question_invalid_output_gives_no_candidates(monkeypatch, reply):
    _paraphrase_with(monkeypatch, reply)
    assert teacher.paraphrase_question("Q?", 3) == []


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated", "timeout", "server"])
def test_paraphrase_question_one_failed_call_gives_no_candidates(monkeypatch, kind):
    _paraphrase_with(monkeypatch, teacher.TeacherAPIError(kind))
    assert teacher.paraphrase_question("Q?", 3) == []


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit", "bad_request"])
def test_paraphrase_question_errors_that_would_repeat_stop_the_run(monkeypatch, kind):
    _paraphrase_with(monkeypatch, teacher.TeacherAPIError(kind, "why"))
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.paraphrase_question("Q?", 3)


# ---- M5: judge_variant ----
VARIANT_Q = "If it takes 1 hour for 60 people to perform an opera, how many hours will it take 600 people?"


def _judge(monkeypatch, *outputs):
    prompts, models = [], []
    queue = list(outputs)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        models.append((model, temperature))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, models


def test_judge_variant_keeps_a_natural_variant_with_the_same_answer(monkeypatch):
    prompts, models = _judge(monkeypatch, '{"natural": true, "answer_unchanged": true}')
    verdict = teacher.judge_variant(_source_row(), VARIANT_Q)
    assert verdict == {"keep": True, "reason": "kept"}
    assert LONG_Q in prompts[0] and VARIANT_Q in prompts[0]
    assert models == [(config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)]


def test_judge_variant_prompt_shows_the_answer_but_not_the_chain(monkeypatch):
    prompts, _ = _judge(monkeypatch, '{"natural": true, "answer_unchanged": true}')
    row = _source_row()
    teacher.judge_variant(row, VARIANT_Q)
    assert row["answer"] in prompts[0]
    assert row["chain"] not in prompts[0]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ('{"natural": false, "answer_unchanged": true}', "not_natural"),
        ('{"natural": true, "answer_unchanged": false}', "answer_changed"),
        ('{"natural": false, "answer_unchanged": false}', "answer_changed"),
    ],
)
def test_judge_variant_drops_variants_the_judge_rejects(monkeypatch, reply, reason):
    _judge(monkeypatch, reply)
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": reason,
    }


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "",
        "[]",
        '{"natural": true}',
        '{"answer_unchanged": true}',
        '{"natural": "yes", "answer_unchanged": true}',
        '{"natural": true, "answer_unchanged": 1}',
        '{"natural": null, "answer_unchanged": null}',
    ],
)
def test_judge_variant_invalid_or_non_boolean_output_is_dropped(monkeypatch, reply):
    _judge(monkeypatch, reply)
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": "invalid_judge_output",
    }


def test_judge_variant_accepts_json_wrapped_in_code_fences(monkeypatch):
    _judge(monkeypatch, '```json\n{"natural": true, "answer_unchanged": true}\n```')
    assert teacher.judge_variant(_source_row(), VARIANT_Q)["keep"] is True


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated", "timeout", "server"])
def test_judge_variant_one_failed_call_drops_only_that_variant(monkeypatch, kind):
    _judge(monkeypatch, teacher.TeacherAPIError(kind))
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": "judge_error",
    }


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit", "bad_request"])
def test_judge_variant_errors_that_would_repeat_stop_the_run(monkeypatch, kind):
    _judge(monkeypatch, teacher.TeacherAPIError(kind, "why"))
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.judge_variant(_source_row(), VARIANT_Q)


def test_build_eval_data_writes_kept_variants_with_their_verdict(tmp_path, monkeypatch):
    rows = [_qrow(1), _qrow(2)]
    summary, _ = _build(tmp_path, monkeypatch, rows)
    out, review, _ = _eval_paths(tmp_path)
    written = data.read_jsonl(out)
    assert [r["id"] for r in written] == ["t-1-v0", "t-1-v1", "t-2-v0", "t-2-v1"]
    assert all(r["verdict"] == {"keep": True, "reason": "kept"} for r in written)
    assert all(r["chain"] == rows[0]["chain"] for r in written[:2])
    assert (summary["kept"], summary["rejected"]) == (4, 0)
    assert data.read_jsonl(review) == []


def test_a_second_run_spends_nothing_and_gives_the_same_files(tmp_path, monkeypatch):
    rows = [_qrow(1), _qrow(2)]
    _build(tmp_path, monkeypatch, rows)
    out, review, _ = _eval_paths(tmp_path)
    first = (out.read_text(encoding="utf-8"), review.read_text(encoding="utf-8"))
    summary, spend = _build(tmp_path, monkeypatch, rows)
    assert spend.asked == []
    assert spend.judged == []
    assert (summary["paraphrase_calls"], summary["judge_calls"]) == (0, 0)
    assert (
        out.read_text(encoding="utf-8"),
        review.read_text(encoding="utf-8"),
    ) == first


def test_build_eval_data_asks_for_a_pool_larger_than_the_kept_target(
    tmp_path, monkeypatch
):
    _, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=2)
    assert spend.asked == [(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)]


def test_build_eval_data_stops_judging_a_question_once_it_has_enough(
    tmp_path, monkeypatch
):
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=2)
    assert len(spend.judged) == 2
    assert summary["judge_calls"] == 2
    assert summary["paraphrase_calls"] == 1


def test_build_eval_data_keeps_judging_past_rejections_until_enough(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    bad = {"keep": False, "reason": "not_natural"}
    spend = _Spend({pool[0]: bad, pool[1]: bad})
    _, spend = _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=2)
    out, review, _ = _eval_paths(tmp_path)
    assert len(spend.judged) == 4
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v2", "t-1-v3"]
    assert [r["id"] for r in data.read_jsonl(review)] == ["t-1-v0", "t-1-v1"]


def test_a_paraphraser_that_repeats_itself_cannot_loop(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, [_qrow(1)], per_question=1)
    spend = _Spend()
    spend.paraphrase = lambda question, n: _cands(question, n)
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=9)
    assert summary["paraphrase_calls"] == 1


def test_a_changed_source_question_stops_before_any_call(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, [_qrow(1)])
    spend = _Spend()
    with pytest.raises(ValueError, match="t-1"):
        _build(
            tmp_path,
            monkeypatch,
            [_trow(1, question="A completely different question text?")],
            spend,
        )
    assert spend.asked == [] and spend.judged == []


def test_build_eval_data_warns_about_a_question_with_no_kept_variants(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[1], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    spend = _Spend({q: reject for q in pool})
    with pytest.warns(UserWarning, match="t-2"):
        summary, _ = _build(tmp_path, monkeypatch, [_qrow(1), _qrow(2)], spend)
    assert summary["questions_without_variants"] == ["t-2"]


def test_build_eval_data_warns_about_a_category_with_no_kept_variants(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[1], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    spend = _Spend({q: reject for q in pool})
    rows = [_qrow(1), _qrow(2, category="false_premise")]
    with pytest.warns(UserWarning, match="false_premise"):
        summary, _ = _build(tmp_path, monkeypatch, rows, spend)
    assert summary["categories_without_variants"] == ["false_premise"]


def test_build_eval_data_no_warning_when_every_question_has_a_variant(
    tmp_path, monkeypatch, recwarn
):
    _build(tmp_path, monkeypatch, [_qrow(1)])
    assert len(recwarn) == 0


def test_a_paraphraser_that_returns_nothing_warns(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="t-1"):
        summary, spend = _build(
            tmp_path, monkeypatch, [_qrow(1)], paraphrase=lambda question, n: []
        )
    assert spend.judged == []
    assert summary["questions_without_variants"] == ["t-1"]


def test_build_eval_data_skips_short_questions_without_calling_the_paraphraser(
    tmp_path, monkeypatch
):
    short = _trow(1, question="Was Talos alive?")
    with pytest.warns(UserWarning, match="t-1"):
        _, spend = _build(tmp_path, monkeypatch, [short])
    assert spend.asked == []


@pytest.mark.parametrize("stage", ["judge", "paraphrase"])
def test_a_stopping_error_keeps_the_store_progress_and_writes_no_outputs(
    tmp_path, monkeypatch, stage
):
    def boom(*args):
        raise teacher.TeacherAPIError("quota", "daily limit")

    spend = _Spend()
    if stage == "judge":
        calls = {"n": 0}

        def judge_then_boom(source_row, variant_question):
            calls["n"] += 1
            if calls["n"] == 2:
                boom()
            return {"keep": True, "reason": "kept"}

        monkeypatch.setattr(teacher, "judge_variant", judge_then_boom)
        paraphrase = spend.paraphrase
    else:
        monkeypatch.setattr(teacher, "judge_variant", spend.judge)
        paraphrase = boom
    out, review, store = _eval_paths(tmp_path)
    with pytest.raises(teacher.TeacherAPIError, match="daily limit"):
        teacher.build_eval_data(
            [_qrow(1)], out, review, store, per_question=2, paraphrase=paraphrase
        )
    assert not out.exists()
    assert not review.exists()
    if stage == "judge":
        record = data.read_jsonl(store)[0]
        assert record["candidates"][0]["verdict"] == {"keep": True, "reason": "kept"}


def test_build_eval_data_progress_ticks_for_every_candidate_it_goes_through(
    tmp_path, monkeypatch
):
    ticks = []
    spend = _Spend()
    monkeypatch.setattr(teacher, "judge_variant", spend.judge)
    out, review, store = _eval_paths(tmp_path)
    for _ in range(2):
        teacher.build_eval_data(
            [_qrow(1)], out, review, store, per_question=2,
            progress=lambda: ticks.append(1), paraphrase=spend.paraphrase,
        )  # fmt: skip
    assert len(ticks) == 4


# ---- rate limits: wait for the delay the server asks for ----
def _rate_limit_error(delay):
    details = [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
    ]
    return _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", details)


@pytest.mark.parametrize(
    ("delay", "seconds"), [("23s", 23.0), ("1.5s", 1.5), ("81429s", 81429.0)]
)
def test_call_model_reads_the_retry_delay_the_server_asks_for(
    monkeypatch, delay, seconds
):
    _use_client(monkeypatch, _rate_limit_error(delay))
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == "rate_limit"
    assert caught.value.retry_after == seconds


def test_call_model_without_retry_info_has_no_retry_after(monkeypatch):
    _use_client(monkeypatch, _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.retry_after is None


def test_generate_waits_the_servers_delay_when_it_is_longer_than_the_default(
    monkeypatch,
):
    _, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError("rate_limit", retry_after=23.0), "fine"]
    )
    assert teacher._generate("p", "m", 0.0) == "fine"
    assert sleeps == [23.0]


def test_generate_uses_the_default_wait_when_the_servers_delay_is_shorter(monkeypatch):
    _, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError("rate_limit", retry_after=0.1), "fine"]
    )
    teacher._generate("p", "m", 0.0)
    assert sleeps == [config.TEACHER_RETRY_SECONDS]


def test_generate_does_not_wait_when_the_delay_is_longer_than_the_maximum(monkeypatch):
    too_long = config.TEACHER_MAX_WAIT_SECONDS + 1
    calls, sleeps = _script_call_model(
        monkeypatch,
        [teacher.TeacherAPIError("rate_limit", retry_after=too_long), "never"],
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.retry_after == too_long
    assert len(calls) == 1
    assert sleeps == []


def test_generate_waits_exactly_the_maximum_delay_allowed(monkeypatch):
    _, sleeps = _script_call_model(
        monkeypatch,
        [
            teacher.TeacherAPIError(
                "rate_limit", retry_after=config.TEACHER_MAX_WAIT_SECONDS
            ),
            "fine",
        ],
    )
    teacher._generate("p", "m", 0.0)
    assert sleeps == [config.TEACHER_MAX_WAIT_SECONDS]


# ---- timing of build_training_data (only rows that were generated count) ----
def test_build_training_data_summary_has_row_seconds(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert summary["row_seconds"] == {"n": 4, "mean": 1.0, "max": 1.0}


def test_build_training_data_row_seconds_count_only_generated_rows(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert summary["row_seconds"] == {"n": 2, "mean": 1.0, "max": 1.0}


def test_build_training_data_row_seconds_when_nothing_was_generated(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["row_seconds"] == {"n": 0, "mean": None, "max": None}


def test_build_training_data_timing_covers_only_this_run(tmp_path, monkeypatch):
    _script_call_model(monkeypatch, ["old call"])
    teacher._generate("p", "m", 0.0)
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["timing"] == {}


def test_build_training_data_timing_reports_the_api_calls_it_made(
    tmp_path, monkeypatch
):
    _script_call_model(monkeypatch, ["fine", "fine"])

    def fake_generate_chain(row):
        teacher._generate("p", "m-1", 0.0)
        return {**row, "chain": "c"}

    monkeypatch.setattr(teacher, "generate_chain", fake_generate_chain)
    out, store = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["timing"]["m-1"]["calls"] == 2
    assert summary["timing"]["m-1"]["outcomes"] == {"ok": 2}


# ---- M7: make_generate_fn (a numpy fake stands in for the model and tokenizer) ----
class _FakeTokenizer:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self.decoded = []
        self.sides = []

    def __call__(self, prompts, return_tensors=None, padding=True):
        self.sides.append(self.padding_side)
        ids = [[ord(c) % 200 + 1 for c in prompt] for prompt in prompts]
        width = max(len(row) for row in ids)
        pad = lambda row: (
            ([0] * (width - len(row)) + row)
            if self.padding_side == "left"
            else (row + [0] * (width - len(row)))
        )
        input_ids = np.array([pad(row) for row in ids])
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).astype(int)}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        self.decoded.append(np.array(token_ids))
        return [self.scripted.pop(0) for _ in range(len(token_ids))]


class _FakeModel:
    device = "cpu"

    def __init__(self, new_tokens=4):
        self.new_tokens = new_tokens
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        new = np.full((input_ids.shape[0], self.new_tokens), 7)
        return np.concatenate([input_ids, new], axis=1)


def test_generate_fn_returns_one_output_per_prompt_in_order():
    tok = _FakeTokenizer(["A", "B", "C"])
    generate = evaluation.make_generate_fn(
        _FakeModel(), tok, max_new_tokens=5, batch_size=8
    )
    assert generate(["one", "three", "five!"]) == ["A", "B", "C"]


def test_generate_fn_splits_into_batches_and_keeps_the_order():
    tok = _FakeTokenizer(list("abcde"))
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=2)
    assert generate(["p1", "p2", "p3", "p4", "p5"]) == list("abcde")
    assert [call["input_ids"].shape[0] for call in model.calls] == [2, 2, 1]


def test_generate_fn_with_a_batch_larger_than_the_data_makes_one_call():
    tok = _FakeTokenizer(["x", "y"])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=64)
    generate(["a", "b"])
    assert len(model.calls) == 1


def test_generate_fn_with_no_prompts_makes_no_calls():
    tok = _FakeTokenizer([])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=2)
    assert generate([]) == []
    assert model.calls == []


def test_generate_fn_pads_on_the_left_so_different_lengths_share_a_batch():
    tok = _FakeTokenizer(["x", "y"])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=8)
    generate(["short", "a much longer prompt"])
    assert tok.sides == ["left"]
    mask = model.calls[0]["attention_mask"]
    assert mask[0][0] == 0 and mask[0][-1] == 1  # the short prompt is padded first
    assert mask[1][0] == 1


def test_generate_fn_decodes_only_the_new_tokens_never_the_prompt():
    tok = _FakeTokenizer(["x"])
    generate = evaluation.make_generate_fn(
        _FakeModel(new_tokens=4), tok, max_new_tokens=5, batch_size=8
    )
    generate(["a prompt of some length"])
    assert tok.decoded[0].shape == (1, 4)
    assert (tok.decoded[0] == 7).all()


def test_generate_fn_is_greedy_and_passes_the_token_cap():
    tok = _FakeTokenizer(["x"])
    model = _FakeModel()
    evaluation.make_generate_fn(model, tok, max_new_tokens=123, batch_size=8)(["p"])
    call = model.calls[0]
    assert call["max_new_tokens"] == 123
    assert call["do_sample"] is False


def test_generate_fn_uses_the_configured_defaults():
    tok = _FakeTokenizer(["x"])
    model = _FakeModel()
    evaluation.make_generate_fn(model, tok)(["p"])
    assert model.calls[0]["max_new_tokens"] == config.EVAL_MAX_NEW_TOKENS


def test_generate_fn_gives_a_pad_token_when_the_tokenizer_has_none():
    tok = _FakeTokenizer(["x"])
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.pad_token == tok.eos_token


def test_generate_fn_cuts_each_output_at_its_final_answer_line():
    tok = _FakeTokenizer([_teacher_chain("9") + "\n\nQuestion: next one?"])
    generate = evaluation.make_generate_fn(
        _FakeModel(), tok, max_new_tokens=5, batch_size=8
    )
    assert generate(["p"]) == [_teacher_chain("9")]


def test_generate_fn_gives_identical_outputs_on_a_repeated_greedy_run():
    first = evaluation.make_generate_fn(
        _FakeModel(), _FakeTokenizer(["same", "same"]), max_new_tokens=5, batch_size=8
    )
    second = evaluation.make_generate_fn(
        _FakeModel(), _FakeTokenizer(["same", "same"]), max_new_tokens=5, batch_size=8
    )
    assert first(["p", "q"]) == second(["p", "q"])


def test_the_teacher_and_the_baseline_share_one_instruction_text():
    assert config.CHAIN_INSTRUCTIONS in teacher.chain_prompt("Q?", "a")
    assert config.CHAIN_INSTRUCTIONS in evaluation.baseline_prompt(_grade_row())


# ---- build_training_data without generating: use only the chains already stored ----
def test_no_generate_writes_only_stored_rows_and_makes_no_calls(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(6), out, store, generate=False)
    assert seen == []
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]
    assert (summary["written"], summary["pending"], summary["generated"]) == (4, 2, 0)
    assert summary["timing"] == {}


def test_no_generate_never_touches_the_store(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    before = store.read_text(encoding="utf-8")
    teacher.build_training_data(_train_rows(5), out, store, generate=False)
    assert store.read_text(encoding="utf-8") == before


def test_no_generate_with_nothing_stored_raises_and_writes_nothing(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    seen = _fake_generate_chain(monkeypatch)
    with pytest.raises(ValueError, match="no stored chains"):
        teacher.build_training_data(_train_rows(3), out, store, generate=False)
    assert seen == []
    assert not out.exists()
    assert not store.exists()


def test_no_generate_still_stops_for_a_changed_row(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    changed = _train_rows(3)
    changed[0] = {**changed[0], "question": "A different question entirely?"}
    with pytest.raises(ValueError, match="r-0"):
        teacher.build_training_data(changed, out, store, generate=False)


def test_no_generate_reports_failed_rows_and_skips_them(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    teacher.build_training_data(_train_rows(3), out, store)
    summary = teacher.build_training_data(_train_rows(3), out, store, generate=False)
    assert summary["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]
    assert _ids(out) == ["r-0", "r-2"]


def test_a_normal_run_has_nothing_pending(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    assert teacher.build_training_data(_train_rows(3), out, store)["pending"] == 0


def test_no_generate_progress_still_ticks_once_per_row(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    ticks = []
    teacher.build_training_data(
        _train_rows(5), out, store, generate=False, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 5


def test_generate_fn_puts_the_padding_side_back_so_training_is_not_affected():
    tok = _FakeTokenizer(["x"])
    tok.padding_side = "right"
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.padding_side == "right"


def test_generate_fn_puts_the_padding_side_back_even_if_generation_throws():
    class _Boom(_FakeModel):
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    tok = _FakeTokenizer(["x"])
    tok.padding_side = "right"
    generate = evaluation.make_generate_fn(_Boom(), tok, max_new_tokens=5, batch_size=8)
    with pytest.raises(RuntimeError, match="generation failed"):
        generate(["p"])
    assert tok.padding_side == "right"


def test_generate_fn_restores_a_left_padding_side_too():
    tok = _FakeTokenizer(["x"])
    tok.padding_side = "left"
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.padding_side == "left"
