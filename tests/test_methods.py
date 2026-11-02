# ---- Method B: prompt and classification ----
import json

import pytest

from src import config, methods
from src.schema import WebPage

GOOD_JSON = (
    '{"title": "t", "body": [{"type": "heading", "level": 1, "text": "h"},'
    ' {"type": "paragraph", "parts": [{"type": "text", "text": "p"}]}]}'
)
GOOD_HTML = "<html><head><title>t</title></head><body><h1>h</h1></body></html>"


def test_method_b_prompt_is_method_a_prompt_plus_the_schema():
    request = config.PROMPTS[0]
    prompt = methods.build_prompt(request)
    assert prompt.startswith(config.BASE_PROMPT_TEMPLATE.format(request=request))
    assert json.dumps(WebPage.model_json_schema()) in prompt
    assert config.SCHEMA_INSTRUCTION in prompt


@pytest.mark.parametrize(
    ("text", "hit_cap", "bucket"),
    [
        (GOOD_JSON, False, "valid"),
        (f"  {GOOD_JSON}\n", False, "valid"),  # surrounding whitespace is harmless
        (
            GOOD_JSON,
            True,
            "valid",
        ),  # complete output is valid even if the cap was reached
        (
            f"Sure! Here is the page: {GOOD_JSON}",
            False,
            "malformed",
        ),  # prose around the JSON
        (f"```json\n{GOOD_JSON}\n```", False, "malformed"),  # markdown fence (strict)
        (GOOD_JSON[:-1], False, "malformed"),  # unclosed bracket, model stopped
        (GOOD_JSON[:40], True, "truncated"),  # cut off by the cap
        ("", True, "truncated"),
        ("", False, "malformed"),
        (GOOD_JSON.replace('"level": 1', '"level": 4'), False, "malformed"),
        ('{"title": "t", "body": []}', False, "malformed"),  # valid JSON, empty body
        (
            GOOD_JSON.replace('"title": "t"', '"title": "t", "extra": 1'),
            False,
            "malformed",
        ),
    ],
)
def test_classify_json_buckets(text, hit_cap, bucket):
    assert methods.classify_json(text, hit_cap)[0] == bucket


@pytest.mark.parametrize(
    ("text", "hit_cap", "bucket"),
    [
        (GOOD_HTML, False, "valid"),
        (GOOD_HTML[:-7], False, "malformed"),  # unclosed, model stopped
        (GOOD_HTML[:-7], True, "truncated"),  # unclosed because the cap cut it off
        (f"Sure! {GOOD_HTML}", False, "malformed"),
        (GOOD_HTML, True, "valid"),
    ],
)
def test_classify_html_buckets(text, hit_cap, bucket):
    assert methods.classify_html(text, hit_cap)[0] == bucket


def test_a_reason_is_given_for_every_non_valid_bucket():
    assert methods.classify_json(GOOD_JSON, False)[1] == ""
    assert methods.classify_json("{", False)[1] != ""
    assert methods.classify_html(GOOD_HTML[:-7], True)[1] != ""


# ---- Method A on Guidance's Mock model: the stream loop, the prompt slicing and the cap ----

from guidance import models


class CharTokenizer:
    """One token per character, so token counts are easy to predict."""

    def encode(self, text, add_special_tokens=False):
        return list(text)


@pytest.fixture
def mock_model(monkeypatch):
    monkeypatch.setattr(
        methods.models,
        "Transformers",
        lambda model, tokenizer, echo: models.Mock(echo=False),
    )


def test_run_guidance_returns_the_whole_page_without_the_prompt(
    mock_model, monkeypatch
):
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 10**6)
    out = methods.run_guidance(None, CharTokenizer(), "PROMPT TEXT:")
    assert out["text"].startswith("<html>") and out["text"].rstrip().endswith("</html>")
    assert "PROMPT TEXT" not in out["text"]
    assert out["hit_cap"] is False and out["new_tokens"] == len(out["text"])


def test_run_guidance_stops_at_the_cap_and_says_so(mock_model, monkeypatch):
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 100)
    out = methods.run_guidance(None, CharTokenizer(), "PROMPT TEXT:")
    assert out["hit_cap"] is True and out["new_tokens"] >= 100
    assert not out["text"].rstrip().endswith("</html>")


def test_a_page_that_finishes_exactly_at_the_cap_is_not_truncated(
    mock_model, monkeypatch
):
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", 10**6)
    length = len(methods.run_guidance(None, CharTokenizer(), "PROMPT TEXT:")["text"])
    monkeypatch.setattr(config, "MAX_NEW_TOKENS", length)
    out = methods.run_guidance(None, CharTokenizer(), "PROMPT TEXT:")
    assert out["hit_cap"] is False and out["text"].rstrip().endswith("</html>")


def test_json_to_html_gives_the_assembled_page_only_for_valid_json():
    from src.schema import to_html

    assert methods.json_to_html(GOOD_JSON) == to_html(
        WebPage.model_validate_json(GOOD_JSON)
    )
    assert methods.json_to_html("not json") is None
    assert methods.json_to_html(f"```json\n{GOOD_JSON}\n```") is None
