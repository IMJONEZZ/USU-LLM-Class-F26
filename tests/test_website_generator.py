"""Backend contract tests use a fake HTTP transport, never a fake GPU claim."""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.website_generator.backend import app as module

PROJECT = Path(module.__file__).resolve().parents[1]


@pytest.fixture
def client():
    with TestClient(module.app) as client:
        yield client


def mock_response(
    monkeypatch,
    *,
    html="<div><h1>Hello</h1><p>Welcome.</p></div>",
    reason="stop",
    status=200,
    error=None,
):
    async def post(url, *, json):
        assert url == "http://127.0.0.1:8000/v1/chat/completions"
        assert (
            json["structured_outputs"]["grammar"]
            == (PROJECT / "backend/grammar.gbnf").read_text()
        )
        assert json["messages"][-1]["content"] == "A book club"
        if error:
            raise error
        return httpx.Response(
            status,
            request=httpx.Request("POST", url),
            json={"choices": [{"finish_reason": reason, "message": {"content": html}}]},
        )

    monkeypatch.setattr(module.app.state.client, "post", post)


def test_generate(client, monkeypatch):
    mock_response(monkeypatch)
    result = client.post("/generate", json={"prompt": "A book club"})
    assert result.status_code == 200
    assert result.json()["html"].startswith("<div><h1>")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": " "},
        {"prompt": "x" * 4001},
        {"prompt": "Hi", "grammar": "override"},
    ],
)
def test_invalid_request(client, body):
    assert client.post("/generate", json=body).status_code == 422


@pytest.mark.parametrize(
    "html",
    [
        "<script>alert(1)</script>",
        '<p onclick="x">Hello</p>',
        '<p><a href="javascript:alert(1)">x</a></p>',
        '<div style="color:red">x</div>',
        "<div><p>unfinished",
        "<p><div>x</div></p>",
        "<ul></ul>",
        "<p>&#106;avascript</p>",
        "<!--comment--><p>x</p>",
        "<!doctype html><p>x</p>",
        '<p><a href="https://bad..com">x</a></p>',
        "<p>&amp</p>",
        "<p>x</p garbage>",
        "<p>x</P>",
        "<p>\x00</p>",
        "<li>x</li>",
        "<div><strong>x</strong></div>",
        "",
        "<p/>text",
        "<p><?x?></p>",
    ],
)
def test_reject_unsafe_html(html):
    with pytest.raises(ValueError):
        module.sanitize_html(html)


def test_nested_safe_html():
    html = (
        "<section><article><h1>Books</h1><p><strong>Read <em>more</em></strong> "
        '<a href="https://example.org/books">Visit</a> &amp; enjoy.</p>'
        "<ul><li>One<ul><li>Two</li></ul></li></ul></article></section>"
    )
    assert module.sanitize_html(html) == html


@pytest.mark.parametrize(
    "options,code",
    [
        ({"reason": "length"}, 502),
        ({"reason": "error"}, 502),
        ({"html": "<script>x</script>"}, 502),
        ({"html": None}, 502),
        ({"status": 500}, 502),
        ({"error": httpx.ConnectError("offline")}, 502),
        ({"error": httpx.ReadTimeout("timeout")}, 504),
    ],
)
def test_inference_errors(client, monkeypatch, options, code):
    mock_response(monkeypatch, **options)
    result = client.post("/generate", json={"prompt": "A book club"})
    assert result.status_code == code
    assert "detail" in result.json()


def test_frontend(client):
    assert 'sandbox=""' in client.get("/").text
    assert client.get("/static/script.js").status_code == 200
