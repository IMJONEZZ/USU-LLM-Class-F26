import socket

import httpx
import pytest
from google import genai
from google.genai import errors, types

from src import config, gemini

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
    monkeypatch.setattr(gemini, "get_client", lambda: client)
    return client


def test_get_client_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=config.API_KEY_ENV):
        gemini.get_client()


def test_get_client_with_key_builds_a_client_without_network(monkeypatch):
    monkeypatch.setenv(config.API_KEY_ENV, "test-key")

    def boom(*a, **k):
        raise AssertionError("network touched while building the client")

    monkeypatch.setattr(socket.socket, "connect", boom)
    assert isinstance(gemini.get_client(), genai.Client)


def test_call_model_returns_text_and_passes_settings(monkeypatch):
    client = _use_client(monkeypatch, _response("hello"))
    assert gemini._call_model("m-1", "the prompt", 0.5) == "hello"
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
    with pytest.raises(gemini.TeacherAPIError) as caught:
        gemini._call_model("m", "p", 0.0)
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
    with pytest.raises(gemini.TeacherAPIError) as caught:
        gemini._call_model("m", "p", 0.0)
    assert caught.value.kind == kind


def test_retry_after_is_read_from_the_error_text():
    assert gemini._retry_after("... 'retryDelay': '12s' ...") == 12.0
    assert gemini._retry_after("no delay here") is None


def _script_call_model(monkeypatch, outcomes):
    calls, sleeps = [], []
    queue = list(outcomes)

    def fake(model, prompt, temperature):
        calls.append(prompt)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gemini, "_call_model", fake)
    monkeypatch.setattr(gemini.time, "sleep", sleeps.append)
    return calls, sleeps


def test_generate_retries_transient_errors_then_succeeds(monkeypatch):
    calls, sleeps = _script_call_model(
        monkeypatch,
        [
            gemini.TeacherAPIError("rate_limit"),
            gemini.TeacherAPIError("timeout"),
            "fine",
        ],
    )
    assert gemini._generate("p", "m", 0.0) == "fine"
    assert len(calls) == 3
    assert sleeps == [config.TEACHER_RETRY_SECONDS] * 2


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "server"])
def test_generate_gives_up_after_max_retries(monkeypatch, kind):
    attempts = 1 + config.TEACHER_MAX_RETRIES
    calls, _ = _script_call_model(
        monkeypatch, [gemini.TeacherAPIError(kind)] * attempts
    )
    with pytest.raises(gemini.TeacherAPIError) as caught:
        gemini._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == attempts


@pytest.mark.parametrize(
    "kind", ["auth", "quota", "bad_request", "empty", "blocked", "truncated"]
)
def test_generate_does_not_retry_permanent_errors(monkeypatch, kind):
    calls, sleeps = _script_call_model(
        monkeypatch, [gemini.TeacherAPIError(kind), "never reached"]
    )
    with pytest.raises(gemini.TeacherAPIError) as caught:
        gemini._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == 1
    assert sleeps == []


def test_generate_does_not_wait_when_the_server_asks_for_too_long(monkeypatch):
    too_long = config.TEACHER_MAX_WAIT_SECONDS + 1
    calls, sleeps = _script_call_model(
        monkeypatch,
        [gemini.TeacherAPIError("rate_limit", retry_after=too_long), "never reached"],
    )
    with pytest.raises(gemini.TeacherAPIError):
        gemini._generate("p", "m", 0.0)
    assert len(calls) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "wrapped", ['{"a": 1}', '```json\n{"a": 1}\n```', '  ```\n{"a": 1}```  ']
)
def test_unwrap_strips_code_fences_and_whitespace(wrapped):
    assert gemini._unwrap(wrapped) == '{"a": 1}'
