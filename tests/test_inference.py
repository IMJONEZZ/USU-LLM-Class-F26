import pytest
import requests

from src import inference
from src.inference import InferenceError

CHAT_REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "3 Ls"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: None)


def script_requests(monkeypatch, outcomes):
    """Make requests.request return or raise each outcome in turn."""
    calls = []
    queue = list(outcomes)

    def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(inference.requests, "request", fake_request)
    return calls


def test_build_headers_requires_a_key():
    with pytest.raises(InferenceError, match="API key"):
        inference.build_headers("")
    assert inference.build_headers("k")["Authorization"] == "Bearer k"


def test_build_payload_is_greedy_by_default():
    payload = inference.build_payload("a6", [{"role": "user", "content": "hi"}])
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 512
    assert payload["model"] == "a6"


def test_chat_returns_text_usage_and_latency(monkeypatch):
    calls = script_requests(monkeypatch, [FakeResponse(200, CHAT_REPLY)])
    result = inference.ask("How many Ls?", api_key="k", base_url="http://x/")
    assert result["text"] == "3 Ls"
    assert result["usage"]["completion_tokens"] == 3
    assert result["latency_seconds"] >= 0
    assert calls[0]["url"] == "http://x/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer k"
    assert calls[0]["json"]["messages"] == [{"role": "user", "content": "How many Ls?"}]


def test_server_errors_are_retried_then_succeed(monkeypatch):
    calls = script_requests(
        monkeypatch,
        [FakeResponse(503), requests.Timeout("slow"), FakeResponse(200, CHAT_REPLY)],
    )
    result = inference.ask("q", api_key="k")
    assert result["text"] == "3 Ls"
    assert len(calls) == 3


def test_gives_up_after_all_retries(monkeypatch):
    calls = script_requests(monkeypatch, [requests.ConnectionError("down")] * 3)
    with pytest.raises(InferenceError, match="Gave up after 3 attempts"):
        inference.ask("q", api_key="k")
    assert len(calls) == 3


def test_wrong_key_is_not_retried(monkeypatch):
    calls = script_requests(monkeypatch, [FakeResponse(401, text="Unauthorized")])
    with pytest.raises(InferenceError, match="HTTP 401"):
        inference.ask("q", api_key="bad")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, body=None),
        FakeResponse(200, body={"choices": []}),
        FakeResponse(200, body={"unexpected": True}),
    ],
)
def test_malformed_reply_raises(monkeypatch, response):
    script_requests(monkeypatch, [response])
    with pytest.raises(InferenceError, match="Unexpected reply"):
        inference.ask("q", api_key="k")


def test_list_models(monkeypatch):
    body = {"data": [{"id": "base"}, {"id": "a6"}]}
    calls = script_requests(monkeypatch, [FakeResponse(200, body)])
    assert inference.list_models(api_key="k", base_url="http://x") == ["base", "a6"]
    assert calls[0]["method"] == "GET"
    assert calls[0]["url"] == "http://x/v1/models"


def test_list_models_malformed_reply(monkeypatch):
    script_requests(monkeypatch, [FakeResponse(200, body={"oops": 1})])
    with pytest.raises(InferenceError, match="Unexpected reply"):
        inference.list_models(api_key="k")


def test_main_prints_answer_and_hides_the_key(monkeypatch, capsys):
    monkeypatch.setenv("VLLM_API_KEY", "super-secret")
    monkeypatch.setenv("A7_SERVER_URL", "http://x")
    script_requests(monkeypatch, [FakeResponse(200, CHAT_REPLY)])
    assert inference.main(["How many Ls?"]) == 0
    output = capsys.readouterr().out
    assert "answer: 3 Ls" in output
    assert "Bearer ****" in output
    assert "super-secret" not in output


def test_main_reports_errors(monkeypatch, capsys):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert inference.main(["q"]) == 1
    assert "No API key" in capsys.readouterr().out
