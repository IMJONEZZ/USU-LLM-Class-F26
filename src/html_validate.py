"""Second-pass HTML check: whitelist tags, no attrs, balanced nesting.

Guidance already constrains generation, but a grammar match ≠ "safe to show."
This catches anything weird that slips through (or bad fixtures in tests).
"""

from __future__ import annotations

from html.parser import HTMLParser

# Keep this list in sync with what html_grammar.py can emit.
ALLOWED_TAGS = frozenset(
    {
        "html",
        "head",
        "title",
        "body",
        "h1",
        "h2",
        "p",
        "ul",
        "ol",
        "li",
        "strong",
        "em",
        "br",
    }
)

VOID_TAGS = frozenset({"br"})


class HtmlValidationError(ValueError):
    """Markup failed the whitelist / structure checks."""


class _WhitelistParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.saw_tag = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.saw_tag = True
        if tag not in ALLOWED_TAGS:
            raise HtmlValidationError(f"Disallowed tag: <{tag}>")
        # No href/onclick/style — attributes are how HTML gets dangerous fast.
        if attrs:
            raise HtmlValidationError(f"Attributes are not allowed on <{tag}>")
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Guidance emits <br />; only void tags may use this form.
        self.saw_tag = True
        if tag not in ALLOWED_TAGS:
            raise HtmlValidationError(f"Disallowed tag: <{tag}>")
        if tag not in VOID_TAGS:
            raise HtmlValidationError(
                f"<{tag}> may not be self-closing in this grammar"
            )
        if attrs:
            raise HtmlValidationError(f"Attributes are not allowed on <{tag}>")

    def handle_endtag(self, tag: str) -> None:
        if tag not in ALLOWED_TAGS:
            raise HtmlValidationError(f"Disallowed end tag: </{tag}>")
        if tag in VOID_TAGS:
            raise HtmlValidationError(f"</{tag}> is not valid for void tags")
        if not self.stack or self.stack[-1] != tag:
            raise HtmlValidationError(f"Unbalanced or unexpected </{tag}>")
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        # Parser already decoded entities; bare angle brackets here are bad news.
        if "<" in data or ">" in data:
            raise HtmlValidationError("Raw '<' or '>' is not allowed in text nodes")


def validate_html(document: str) -> str:
    """Return *document* unchanged if it passes; otherwise raise."""
    if not document or not document.strip():
        raise HtmlValidationError("HTML document is empty")

    parser = _WhitelistParser()
    try:
        parser.feed(document)
        parser.close()
    except HtmlValidationError:
        raise
    except Exception as exc:  # pragma: no cover - HTMLParser almost never does this
        raise HtmlValidationError(f"Malformed HTML: {exc}") from exc

    if not parser.saw_tag:
        raise HtmlValidationError("HTML document contains no tags")
    if parser.stack:
        raise HtmlValidationError(f"Unclosed tags remain: {'/'.join(parser.stack)}")

    # Cheap shape checks the stack doesn't cover by itself.
    stripped = document.strip()
    if not stripped.lower().startswith("<html"):
        raise HtmlValidationError("Document must start with <html>")
    if "</html>" not in stripped.lower():
        raise HtmlValidationError("Document must include a closing </html>")
    return document
