"""
Tests for src/html_service.py.

Tests focus on validate_html(), which checks that generated HTML is
well-formed and only uses allowed tags -- this is real, GPU-independent
logic with genuine edge cases. The actual model loading and generation
functions require a GPU and a large model download, so they're marked
# pragma: no cover, consistent with how GPU-dependent code has been
handled in every assignment this semester.
"""

from src.html_service import validate_html

# ---------------------------------------------------------------------------
# Valid HTML
# ---------------------------------------------------------------------------


def test_validate_html_accepts_well_formed_simple_page():
    """A simple, well-formed page with one heading and one paragraph should be valid."""
    html = "<html><body><h1>Hello</h1><p>World</p></body></html>"
    is_valid, errors = validate_html(html)
    assert is_valid
    assert errors == []


def test_validate_html_accepts_multiple_paragraphs():
    """A page with multiple paragraphs after the heading should be valid."""
    html = "<html><body><h1>Title</h1><p>First.</p><p>Second.</p><p>Third.</p></body></html>"
    is_valid, errors = validate_html(html)
    assert is_valid
    assert errors == []


# ---------------------------------------------------------------------------
# Disallowed tags
# ---------------------------------------------------------------------------


def test_validate_html_rejects_disallowed_tag():
    """A tag outside the allowed set (html, body, h1, p) should be rejected."""
    html = "<html><body><h1>Hello</h1><script>alert(1)</script></body></html>"
    is_valid, errors = validate_html(html)
    assert not is_valid
    assert any("script" in e for e in errors)


def test_validate_html_rejects_multiple_disallowed_tags():
    """Multiple disallowed tags should each be reported."""
    html = "<html><body><div>Hello</div><span>World</span></body></html>"
    is_valid, errors = validate_html(html)
    assert not is_valid
    assert any("div" in e for e in errors)
    assert any("span" in e for e in errors)


# ---------------------------------------------------------------------------
# Malformed structure
# ---------------------------------------------------------------------------


def test_validate_html_rejects_unclosed_tag():
    """HTML with a tag that's never closed should be rejected."""
    html = "<html><body><h1>Hello</h1><p>World</body></html>"
    is_valid, errors = validate_html(html)
    assert not is_valid
    assert any("Unclosed" in e or "Mismatched" in e for e in errors)


def test_validate_html_rejects_mismatched_closing_tag():
    """A closing tag that doesn't match the most recently opened tag should be rejected."""
    html = "<html><body><h1>Hello</p></body></html>"
    is_valid, errors = validate_html(html)
    assert not is_valid
    assert any("Mismatched" in e for e in errors)


def test_validate_html_rejects_closing_tag_with_no_open_tag():
    """A closing tag with nothing open at all should be rejected, not crash."""
    html = "</p>"
    is_valid, errors = validate_html(html)
    assert not is_valid
    assert any("no matching open tag" in e for e in errors)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_validate_html_accepts_empty_string_as_structurally_valid():
    """
    An empty string has no tags at all, so there's nothing structurally
    wrong with it -- no mismatched tags, no disallowed tags, no unclosed
    tags. validate_html checks structural correctness, not completeness,
    so this correctly returns valid.
    """
    is_valid, errors = validate_html("")
    assert is_valid
    assert errors == []


def test_validate_html_accepts_empty_paragraph_tags():
    """A paragraph tag with no text content should still be structurally valid."""
    html = "<html><body><h1>Title</h1><p></p></body></html>"
    is_valid, _errors = validate_html(html)
    assert is_valid


# ---------------------------------------------------------------------------
# load_model() / generate_html() -- GPU-dependent
# Intentionally NOT tested here, since they require a GPU and downloading
# a 7B model, which this CI environment does not have.
# ---------------------------------------------------------------------------


def test_html_service_module_is_importable_without_a_gpu():
    """
    Confirm that importing src.html_service doesn't itself require a GPU
    or trigger model loading -- only calling load_model()/generate_html()
    should need one.
    """
    import src.html_service  # noqa: F401
