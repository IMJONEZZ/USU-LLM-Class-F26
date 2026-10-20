"""
Tests for src/inference.py.

inference.py sends real HTTP requests to a locally-running llama.cpp
server, which doesn't exist in CI. The HTTP-calling function is injected
as a parameter (http_post), so these tests supply a fake version that
returns a canned response, keeping the test suite fast and runnable
without a live server or GPU.
"""

import pytest
import requests

from src.inference import send_chat_request


class FakeResponse:
    """Mimics a requests.Response object."""

    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def make_fake_post(response_content, status_code=200):
    calls = []

    def fake_post(url, json=None):
        calls.append({"url": url, "json": json})
        fake_json = {
            "choices": [{"message": {"role": "assistant", "content": response_content}}]
        }
        return FakeResponse(fake_json, status_code=status_code)

    fake_post.calls = calls
    return fake_post


def test_send_chat_request_returns_response_content():
    fake_post = make_fake_post("The answer is 42.")
    result = send_chat_request("What is the meaning of life?", http_post=fake_post)
    assert result == "The answer is 42."


def test_send_chat_request_hits_correct_endpoint():
    fake_post = make_fake_post("Some response")
    send_chat_request(
        "A question", server_url="http://example.com", http_post=fake_post
    )
    assert fake_post.calls[0]["url"] == "http://example.com/v1/chat/completions"


def test_send_chat_request_includes_prompt_in_payload():
    fake_post = make_fake_post("Some response")
    send_chat_request("What is 2 plus 2?", http_post=fake_post)
    sent_json = fake_post.calls[0]["json"]
    assert sent_json["messages"][0]["content"] == "What is 2 plus 2?"
    assert sent_json["messages"][0]["role"] == "user"


def test_send_chat_request_includes_max_tokens():
    fake_post = make_fake_post("Some response")
    send_chat_request("A question", max_tokens=75, http_post=fake_post)
    sent_json = fake_post.calls[0]["json"]
    assert sent_json["max_tokens"] == 75


def test_send_chat_request_uses_deterministic_temperature():
    fake_post = make_fake_post("Some response")
    send_chat_request("A question", http_post=fake_post)
    sent_json = fake_post.calls[0]["json"]
    assert sent_json["temperature"] == 0


def test_send_chat_request_raises_on_http_error():
    """An HTTP error response should propagate as an exception, not be swallowed."""
    fake_post = make_fake_post("irrelevant", status_code=500)
    with pytest.raises(requests.HTTPError):
        send_chat_request("A question", http_post=fake_post)


def test_send_chat_request_uses_default_server_url_if_not_specified():
    fake_post = make_fake_post("Some response")
    send_chat_request("A question", http_post=fake_post)
    assert "localhost:8000" in fake_post.calls[0]["url"]
