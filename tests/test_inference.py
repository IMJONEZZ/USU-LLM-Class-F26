"""Tests for the client that talks to the served model.

None of these touch the network. The session is an argument to `ask`, so the
tests hand it a fake that returns whatever response the test is about. That
matters more here than in the other files, because the thing this code talks to
is a container that scales to zero and costs money to wake up.

The retry tests are the ones I care about. A cold server answers 503 for the
first ten seconds or so, and the difference between handling that and not is
the difference between the homework working and the grader seeing a stack
trace on their first request.
"""

import json
import sys

import pytest
import requests

from src.inference import (
    BASE_URL,
    MODEL,
    ServerNotReady,
    api_key,
    ask,
    build_request,
    is_still_loading,
    show,
)


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def completion(text):
    return FakeResponse(200, {"choices": [{"message": {"content": text}}]})


def loading():
    return FakeResponse(
        503, {"error": {"message": "Loading model", "type": "unavailable_error"}}
    )


class FakeSession:
    """Hands back queued responses and records what it was asked to send."""

    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        return self.queue.pop(0) if self.queue else completion("fallback")


# --- the key ------------------------------------------------------------------


def test_api_key_reads_the_environment():
    assert api_key({"LLAMA_API_KEY": "abc123"}) == "abc123"


def test_api_key_complains_clearly_when_it_is_missing():
    # The server answers 401 without it, and "401" on its own is a worse
    # message than saying which variable to set.
    with pytest.raises(RuntimeError, match="LLAMA_API_KEY"):
        api_key({})


def test_api_key_treats_empty_as_missing():
    with pytest.raises(RuntimeError):
        api_key({"LLAMA_API_KEY": ""})


# --- the request body ---------------------------------------------------------


def test_build_request_has_the_shape_the_openai_api_wants():
    body = build_request("hello?")
    assert body["model"] == MODEL
    assert body["messages"] == [{"role": "user", "content": "hello?"}]


def test_build_request_does_not_sample():
    """Temperature has to stay at zero or nothing is comparable.

    The assignment 6 answers were greedy. If this sampled, a difference between
    the two could be the quantization or could be luck, and I couldn't tell.
    """
    assert build_request("q")["temperature"] == 0.0


def test_build_request_token_budget_is_adjustable():
    assert build_request("q", max_tokens=12)["max_tokens"] == 12


# --- telling a cold start apart from a real failure ---------------------------


def test_a_loading_503_is_recognized():
    assert is_still_loading(loading())


def test_a_200_is_not_loading():
    assert not is_still_loading(completion("hi"))


def test_a_503_with_no_json_body_counts_as_loading():
    # Something in front of llama.cpp can return a bare 503 while the container
    # comes up. Treating that as "wait" is the safer guess.
    assert is_still_loading(FakeResponse(503, None, text="Bad Gateway"))


def test_a_500_is_not_treated_as_a_cold_start():
    """A real error should surface, not get retried into a timeout."""
    assert not is_still_loading(FakeResponse(500, {"error": {"message": "boom"}}))


def test_only_a_503_counts_as_loading_even_when_the_body_says_loading():
    """Pins the status check rather than leaning on the body check.

    I broke the status code test on purpose to see whether anything failed and
    nothing did, because the body check was quietly doing all the work. Only
    llama.cpp's 503 means the model is still coming up; a 500 that happens to
    mention loading is a different problem and retrying it just hides it.
    """
    assert not is_still_loading(
        FakeResponse(500, {"error": {"message": "Loading model"}})
    )


# --- asking --------------------------------------------------------------------


def test_ask_returns_the_assistant_message():
    session = FakeSession(completion("  three  "))
    assert ask("q", key="k", session=session) == "three"


def test_ask_sends_the_bearer_token():
    session = FakeSession(completion("ok"))
    ask("q", key="secret", session=session)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer secret"


def test_ask_posts_to_the_chat_completions_route():
    session = FakeSession(completion("ok"))
    ask("q", key="k", session=session)
    assert session.calls[0]["url"] == f"{BASE_URL}/v1/chat/completions"


def test_ask_waits_out_a_cold_start_and_then_succeeds():
    """This is the one that matters.

    Two 503s then an answer is exactly what a scaled-to-zero container does.
    """
    slept = []
    session = FakeSession(loading(), loading(), completion("awake"))
    out = ask("q", key="k", session=session, sleep=slept.append, retry_seconds=0.1)
    assert out == "awake"
    assert len(session.calls) == 3
    assert slept == [0.1, 0.1]


def test_ask_gives_up_rather_than_retrying_forever():
    session = FakeSession(*[loading() for _ in range(4)])
    with pytest.raises(ServerNotReady, match="still loading"):
        ask("q", key="k", session=session, attempts=4, sleep=lambda _: None)
    assert len(session.calls) == 4


def test_ask_does_not_swallow_a_real_http_error():
    # A 401 means the key is wrong. Retrying that just wastes time and then
    # reports the wrong problem.
    session = FakeSession(FakeResponse(401, {"error": {"message": "Invalid API Key"}}))
    with pytest.raises(requests.HTTPError):
        ask("q", key="k", session=session)


def test_ask_falls_back_to_the_environment_for_the_key(monkeypatch):
    monkeypatch.setenv("LLAMA_API_KEY", "from-env")
    session = FakeSession(completion("ok"))
    ask("q", session=session)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer from-env"


# --- the printed output, which is what the screenshot shows -------------------


def test_show_prints_the_request_and_the_response(capsys):
    session = FakeSession(completion("three"))
    show("How many Ls?", key="k", session=session)
    printed = capsys.readouterr().out
    assert "REQUEST" in printed
    assert "RESPONSE" in printed
    assert "How many Ls?" in printed
    assert "three" in printed


def test_show_never_prints_the_key():
    """The screenshot goes on a public PR."""
    session = FakeSession(completion("ok"))
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        show("q", key="super-secret-key", session=session)
    assert "super-secret-key" not in buffer.getvalue()
    assert "Bearer ****" in buffer.getvalue()


def test_the_module_does_not_run_anything_on_import():
    """Importing it shouldn't fire a request at a server that costs money."""
    import importlib

    sys.modules.pop("src.inference", None)
    module = importlib.import_module("src.inference")
    assert module.BASE_URL.startswith("https://")


def test_the_request_body_is_json_serializable():
    # requests will serialize this, so a non-serializable value fails at call
    # time against a live server rather than here.
    json.dumps(build_request("q"))
