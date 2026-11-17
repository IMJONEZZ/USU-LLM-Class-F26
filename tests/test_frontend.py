import json

import pytest

from src import config, frontend, inference


class _FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def iter_lines(self, **kwargs):
        return iter(self.lines)


def _sse(*texts):
    chunks = [{"choices": [{"text": t, "finish_reason": None}]} for t in texts]
    chunks.append({"choices": [], "usage": {"completion_tokens": len(texts)}})
    return ["data: " + json.dumps(c) for c in chunks] + ["", "data: [DONE]"]


def _serve(monkeypatch, lines):
    seen = {}

    def fake_call(method, path, **kwargs):
        seen.update(kwargs["json"])
        return _FakeResponse(lines), 0.0, 0

    monkeypatch.setattr(inference, "call", fake_call)
    return seen


def test_chat_sends_only_the_latest_message_not_the_history(monkeypatch):
    seen = _serve(monkeypatch, _sse("a", "b"))
    history = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second answer"},
    ]
    list(frontend.chat("third question", history))
    assert seen["prompt"] == inference.format_prompt("third question")
    assert seen["model"] == config.ADAPTER_NAME
    assert seen["stream"] is True


def test_chat_yields_the_whole_reply_so_far_and_skips_the_usage_chunk(monkeypatch):
    _serve(monkeypatch, _sse("a", "b", "c"))
    outputs = list(frontend.chat("Q?", []))
    assert outputs[:3] == ["a", "ab", "abc"]
    assert "usage" not in outputs[-1]


def test_chat_adds_one_status_line_after_the_last_text(monkeypatch):
    _serve(monkeypatch, _sse("a", "b", "c"))
    outputs = list(frontend.chat("Q?", []))
    assert len(outputs) == 4
    assert outputs[-1].startswith("abc")
    assert outputs[-1].count("tok/s") == 1
    assert "3 chunks" in outputs[-1]
    assert not any("tok/s" in o for o in outputs[:-1])


@pytest.mark.parametrize("message", ["", "   ", "### Response:\nhi"])
def test_chat_rejects_a_bad_message_before_any_request(monkeypatch, message):
    def no_network(*args, **kwargs):
        raise AssertionError("a request was sent")

    monkeypatch.setattr(inference, "call", no_network)
    with pytest.raises(ValueError):
        next(frontend.chat(message, []))
