import threading

import pytest
import requests

from src import inference


class FakeResponse:
    def __init__(self, text="generated answer", error=None):
        self.text = text
        self.error = error

    def raise_for_status(self):
        if self.error:
            raise self.error

    def json(self):
        return {"choices": [{"text": f" {self.text} "}]}


def test_questions_has_ten_prompts():
    assert len(inference.QUESTIONS) == 10


def test_ask_posts_formatted_question_and_returns_stripped_text(monkeypatch):
    observed = {}

    def fake_post(url, *, json, timeout):
        observed.update(url=url, json=json, timeout=timeout)
        return FakeResponse("  answer text  ")

    monkeypatch.setattr(inference.requests, "post", fake_post)

    result = inference.ask("What is the answer?")

    assert result == "answer text"
    assert observed == {
        "url": inference.API_URL,
        "json": {
            "model": inference.MODEL_NAME,
            "prompt": "Question:\nWhat is the answer?\nAnswer:\n",
            "max_tokens": 512,
            "temperature": 0,
        },
        "timeout": 300,
    }


def test_ask_propagates_http_errors(monkeypatch):
    def fake_post(*_args, **_kwargs):
        return FakeResponse(error=requests.HTTPError("server unavailable"))

    monkeypatch.setattr(inference.requests, "post", fake_post)

    with pytest.raises(requests.HTTPError, match="server unavailable"):
        inference.ask("question")


def test_main_sends_all_requests_concurrently_and_prints_in_question_order(
    monkeypatch, capsys
):
    barrier = threading.Barrier(len(inference.QUESTIONS))
    response_order = []
    lock = threading.Lock()

    def fake_post(_url, *, json, timeout):
        question = json["prompt"].split("\n", 2)[1]
        with lock:
            response_order.append(question)
        barrier.wait(timeout=3)
        return FakeResponse(f"answer for {question}")

    monkeypatch.setattr(inference.requests, "post", fake_post)

    inference.main()

    output = capsys.readouterr().out
    assert len(response_order) == len(inference.QUESTIONS)
    assert all(
        output.index(question) < output.index(f"answer for {question}")
        for question in inference.QUESTIONS
    )
    assert [output.index(question) for question in inference.QUESTIONS] == sorted(
        output.index(question) for question in inference.QUESTIONS
    )


def test_main_can_be_called_with_a_small_executor(monkeypatch, capsys):
    """Exercise main's executor mapping with a deterministic local stub."""
    answers = iter(f"answer-{index}" for index in range(len(inference.QUESTIONS)))
    monkeypatch.setattr(inference, "ask", lambda _question: next(answers))

    inference.main()

    output = capsys.readouterr().out
    assert "1. Question:" in output
    assert "10. Question:" in output
    assert "answer-0" in output
    assert "answer-9" in output
