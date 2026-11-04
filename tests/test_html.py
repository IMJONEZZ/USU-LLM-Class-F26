"""Tests for the HTML service. A fake model stands in for Qwen, so these run in CI.

Why these tests: the risky parts are (1) the validator, which must reject broken
pages (Python's HTMLParser alone accepts anything), (2) the grammar, which must
only allow what the validator accepts, and (3) the HTTP layer.
"""

import json
import os
import re

import pytest
from fastapi.testclient import TestClient

from src import html_service
from src.app import create_app
from src.html_service import (
    ALLOWED_TAGS,
    EVAL_PROMPTS,
    HTML_GRAMMAR,
    OutlinesModel,
    evaluate,
    generate_page,
    main,
    validate_html,
)

GOOD = (
    "<html><head><title>Mars</title></head><body><h1>Mars</h1>"
    "<p>Mars is red.</p><ul><li>Phobos</li><li>Deimos</li></ul></body></html>"
)
TRUNCATED = "<html><head><title>Mars</title></head><body><h1>Mars</h1><ul><li>Pho"


class FakeModel:
    """Returns good HTML when given a grammar and broken HTML when not."""

    def __init__(self, good=GOOD, bad=TRUNCATED):
        self.good, self.bad, self.calls = good, bad, []

    def generate(self, prompt, grammar, max_new_tokens):
        self.calls.append((prompt, grammar, max_new_tokens))
        return self.good if grammar else self.bad


# ---- validator -------------------------------------------------------------


def test_valid_page_has_no_errors():
    assert validate_html(GOOD) == []


def test_surrounding_whitespace_is_ignored():
    assert validate_html("\n  " + GOOD + "  \n") == []


@pytest.mark.parametrize(
    "text, fragment",
    [
        ("", "empty output"),
        ("   ", "empty output"),
        (TRUNCATED, "does not end with </html>"),
        (TRUNCATED, "unclosed tags"),
        ("<p>hi</p>", "does not start with <html>"),
        (GOOD.replace("<h1>Mars</h1>", "<div>Mars</div>"), "tag not allowed: <div>"),
        (GOOD.replace("</p>", "</h1>", 1), "expected </p> but found </h1>"),
        (GOOD + "</p>", "does not end with </html>"),
        ("<html></html></p>", "closing </p> with nothing open"),
    ],
)
def test_validator_rejects_broken_pages(text, fragment):
    assert any(fragment in error for error in validate_html(text))


def test_validator_recovers_after_misnested_close():
    # </h1> closes <h1> and everything opened after it; no crash, error reported.
    text = "<html><body><h1><p>x</h1></body></html>"
    assert validate_html(text)


# ---- grammar ---------------------------------------------------------------


def test_grammar_covers_exactly_the_allowed_tags():
    for tag in ALLOWED_TAGS:
        assert f"<{tag}>" in HTML_GRAMMAR
        assert f"</{tag}>" in HTML_GRAMMAR


def _text_pattern():
    rule = next(line for line in HTML_GRAMMAR.splitlines() if line.startswith("TEXT:"))
    return re.compile(rule.split("/")[1] + "$")


@pytest.mark.parametrize("char", ["<", ">", "&", "\n", "\t", "\x00", "\x7f"])
def test_grammar_text_rejects_markup_and_control_characters(char):
    assert _text_pattern().match("abc" + char + "def") is None


@pytest.mark.parametrize(
    "text",
    ["Mars is red.", "caf\u00e9", "5% of $10", "x = y + 1", "\u65e5\u672c\u8a9e"],
)
def test_grammar_text_allows_everything_else(text):
    assert _text_pattern().match(text)


# ---- generate_page / evaluate ---------------------------------------------


def test_generate_page_constrained_is_valid_and_sends_grammar():
    model = FakeModel()
    result = generate_page(model, "about Mars", constrained=True, max_new_tokens=123)
    assert result["valid"] and result["errors"] == []
    assert result["html"] == GOOD and result["constrained"] is True
    assert model.calls == [("about Mars", HTML_GRAMMAR, 123)]


def test_generate_page_unconstrained_reports_errors():
    model = FakeModel()
    result = generate_page(model, "about Mars", constrained=False)
    assert not result["valid"] and result["errors"]
    assert model.calls[0][1] is None


def test_evaluate_compares_both_modes_on_every_prompt():
    model = FakeModel()
    summary = evaluate(model)
    constrained, unconstrained = (
        summary["runs"]["constrained"],
        summary["runs"]["unconstrained"],
    )
    assert constrained["total"] == unconstrained["total"] == len(EVAL_PROMPTS) == 10
    assert constrained["valid_percent"] == 100.0
    assert unconstrained["valid_percent"] == 0.0
    assert len(model.calls) == 20


def test_evaluate_handles_no_prompts():
    summary = evaluate(FakeModel(), prompts=())
    assert summary["runs"]["constrained"]["valid_percent"] == 0.0
    assert summary["runs"]["constrained"]["mean_seconds"] == 0.0


# ---- OutlinesModel wrapper -------------------------------------------------


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


def test_outlines_wrapper_passes_grammar_through_cfg_type():
    seen = {}

    def model(text, *args, max_new_tokens):
        seen.update(text=text, args=args, max_new_tokens=max_new_tokens)
        return "OUT"

    wrapper = OutlinesModel(model, FakeTokenizer(), lambda g: ("CFG", g))
    assert wrapper.generate("hello", HTML_GRAMMAR, 77) == "OUT"
    assert seen["args"] == (("CFG", HTML_GRAMMAR),)
    assert seen["max_new_tokens"] == 77
    assert "system:" in seen["text"] and "user:hello" in seen["text"]


def test_outlines_wrapper_without_grammar_sends_no_output_type():
    seen = {}

    def model(text, *args, max_new_tokens):
        seen["args"] = args
        return "OUT"

    wrapper = OutlinesModel(model, FakeTokenizer(), lambda g: ("CFG", g))
    assert wrapper.generate("hello", None, 50) == "OUT"
    assert seen["args"] == ()


# ---- HTTP service ----------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(create_app(FakeModel()))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_generate_returns_structured_html(client):
    response = client.post("/generate", json={"prompt": "about Mars"})
    assert response.status_code == 200
    body = response.json()
    assert body["html"] == GOOD and body["valid"] is True and body["errors"] == []
    assert body["constrained"] is True and body["seconds"] >= 0


def test_generate_unconstrained_flag_reaches_the_model(client):
    body = client.post(
        "/generate", json={"prompt": "about Mars", "constrained": False}
    ).json()
    assert body["valid"] is False and body["constrained"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"prompt": ""},
        {"prompt": "x" * 501},
        {"prompt": "ok", "max_new_tokens": 10},
        {"prompt": "ok", "max_new_tokens": 5000},
    ],
)
def test_generate_rejects_bad_requests(client, payload):
    assert client.post("/generate", json=payload).status_code == 422


# ---- command line ----------------------------------------------------------


def test_cli_eval_writes_results(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(html_service, "load_model", lambda: FakeModel())
    out = tmp_path / "results.json"
    assert main(["eval", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["runs"]["constrained"]["valid"] == 10
    assert "constrained: 10/10 valid (100.0%)" in capsys.readouterr().out


# ---- real model (runs only on my Mac, skipped on GitHub) -----------------------


@pytest.mark.skipif(
    not os.environ.get("RUN_REAL_MODEL"),
    reason="slow: loads Qwen2.5-1.5B (~3 GB) on the Apple GPU. Run with RUN_REAL_MODEL=1",
)
def test_real_model_with_grammar_returns_valid_html():
    """The one test that touches the real model, grammar and checker together."""
    pytest.importorskip("outlines")
    pytest.importorskip("torch")
    model = html_service.load_model()
    result = generate_page(model, "Write a small HTML page about the planet Mars.")
    assert result["valid"], result["errors"]
    assert result["html"].startswith("<html>")
