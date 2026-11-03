import pytest
from pydantic import ValidationError

from src.schema import Heading, WebPage


@pytest.mark.parametrize(
    "bad",
    [
        {"body": [{"type": "heading", "level": 1, "text": "x"}]},  # missing title
        {"title": "t", "body": []},  # empty body
        {"title": "t", "body": [{"type": "heading", "level": 4, "text": "x"}]},
        {"title": "t", "body": [{"type": "heading", "level": 1, "text": ""}]},
        {"title": "t", "body": [{"type": "paragraph", "parts": []}]},
        {"title": "t", "body": [{"type": "paragraph", "parts": [{"type": "link"}]}]},
        {"title": "t", "body": [{"type": "table"}]},
        {
            "title": "t",
            "extra": 1,
            "body": [{"type": "paragraph", "parts": [{"type": "br"}]}],
        },
    ],
)
def test_invalid_pages_raise_validation_error(bad):
    with pytest.raises(ValidationError):
        WebPage.model_validate(bad)


def test_valid_json_from_a_model_is_accepted():
    raw = (
        '{"title": "t", "body": [{"type": "heading", "level": 2, "text": "h"},'
        ' {"type": "paragraph", "parts": [{"type": "text", "text": "p"}, {"type": "br"}]}]}'
    )
    page = WebPage.model_validate_json(raw)
    assert page.body[0] == Heading(type="heading", level=2, text="h")


def test_json_schema_marks_type_required_everywhere_validation_requires_it():
    """The schema goes into Method B's prompt: it must not call `type` optional if validation rejects its absence."""
    defs = WebPage.model_json_schema()["$defs"]
    assert all("type" in d["required"] for d in defs.values())
