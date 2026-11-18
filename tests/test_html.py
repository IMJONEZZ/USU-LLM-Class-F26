"""Tests for the fixed HTML output grammar."""

import re
import sys
from types import ModuleType
from unittest.mock import Mock

import outlines
import pytest
from outlines.inputs import Chat

from src.html_generation import (
    HTML_GRAMMAR,
    HTML_OUTPUT_FORMAT,
    SYSTEM_PROMPT,
    HtmlAnswerGenerator,
)

EXPECTED_RULES = {
    "start": (
        "doctype html_open head_open meta title_open title_text title_close "
        "head_close body_open h1_open heading_text h1_close p_open "
        "paragraph_text p_close body_close html_close"
    ),
    "doctype": '"<!doctype html>"',
    "html_open": '"<html lang=\\"en\\">"',
    "head_open": '"<head>"',
    "meta": '"<meta charset=\\"utf-8\\">"',
    "title_open": '"<title>"',
    "title_text": "HTML_TEXT",
    "title_close": '"</title>"',
    "head_close": '"</head>"',
    "body_open": '"<body>"',
    "h1_open": '"<h1>"',
    "heading_text": "HTML_TEXT",
    "h1_close": '"</h1>"',
    "p_open": '"<p>"',
    "paragraph_text": "HTML_TEXT",
    "p_close": '"</p>"',
    "body_close": '"</body>"',
    "html_close": '"</html>"',
    "HTML_TEXT": "/[^<>]+/",
}


def _grammar_rules() -> dict[str, str]:
    """Return the named rules and terminals declared in the grammar."""
    rules = {}
    for line in HTML_GRAMMAR.splitlines():
        name, separator, definition = line.strip().partition(":")
        if separator:
            rules[name] = definition.strip()
    return rules


def test_only_the_three_text_fields_are_variable() -> None:
    assert _grammar_rules() == EXPECTED_RULES


@pytest.mark.parametrize(
    "text",
    ["A title", "A longer heading.", "An answer with & and apostrophes."],
)
def test_html_text_rule_accepts_text_without_angle_brackets(text: str) -> None:
    pattern = re.fullmatch(r"[^<>]+", text)
    assert _grammar_rules()["HTML_TEXT"] == "/[^<>]+/"
    assert pattern is not None


@pytest.mark.parametrize("text", ["<", ">", "before <tag", "after > tag"])
def test_html_text_rule_rejects_angle_brackets(text: str) -> None:
    pattern = re.fullmatch(r"[^<>]+", text)
    assert _grammar_rules()["HTML_TEXT"] == "/[^<>]+/"
    assert pattern is None


def test_required_elements_have_opening_and_closing_tags() -> None:
    rules = _grammar_rules()
    tag_pairs = {
        "html": (rules["html_open"], rules["html_close"]),
        "head": (rules["head_open"], rules["head_close"]),
        "title": (rules["title_open"], rules["title_close"]),
        "body": (rules["body_open"], rules["body_close"]),
        "h1": (rules["h1_open"], rules["h1_close"]),
        "p": (rules["p_open"], rules["p_close"]),
    }

    for tag, (opening, closing) in tag_pairs.items():
        assert opening.startswith('"<') and opening.endswith('>"'), tag
        assert closing.startswith('"</') and closing.endswith('>"'), tag


def test_generate_passes_both_prompts_and_html_grammar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "Why does the sky look blue?"
    expected_html = '<html lang="en">...'
    model = Mock(return_value=expected_html)

    class FakeSamplingParams:
        def __init__(self, max_tokens: int) -> None:
            self.max_tokens = max_tokens

    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = Mock(return_value=object())
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setattr(outlines, "from_vllm_offline", lambda _llm: model)

    result = HtmlAnswerGenerator().generate.local(question)

    assert result == expected_html
    prompt = model.call_args.args[0]
    assert isinstance(prompt, Chat)
    assert prompt.messages == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    assert model.call_args.args[1] is HTML_OUTPUT_FORMAT
    assert model.call_args.kwargs["sampling_params"].max_tokens == 512
