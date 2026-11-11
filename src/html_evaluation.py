"""Shared HTML specification, validation and post-generation scoring for A10 and its A9 reference."""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser

from guidance import gen, guidance, select
from guidance.library import capture, with_temperature
from guidance.library._sequences import repeat

# repeat isn't re-exported from guidance.library.__init__, but this is the
# same helper one_or_more uses under the hood — we need max= so Mock/Qwen
# can't spin forever on one_or_more.


@guidance(stateless=True)
def _gen_text(lm):
    # Block '<' / '>' so the model can't sneak tags into a text span.
    # Also ban newlines so each text chunk stays on one line (easier to read).
    #
    # Guidance gen(max_tokens=...) — original span budget. (Under a regex this
    # limit is soft at decode time; html_grammar().match() is stricter, so the
    # HTTP path trusts constrained decode + whitelist validation.)
    return lm + gen(regex="[^<>\n]+", max_tokens=32)


@guidance(stateless=True)
def _gen_text_in_tag(lm, tag: str):
    # We write the tags ourselves; the model only fills the inside.
    lm += f"<{tag}>"
    lm += _gen_text()
    lm += f"</{tag}>"
    return lm


@guidance(stateless=True)
def _gen_header(lm):
    lm += "<head>\n"
    lm += _gen_text_in_tag("title") + "\n"
    lm += "</head>\n"
    return lm


@guidance(stateless=True)
def _gen_heading(lm):
    # Model chooses h1 vs h2; both are fully wrapped by us.
    lm += select(options=[_gen_text_in_tag("h1"), _gen_text_in_tag("h2")])
    lm += "\n"
    return lm


@guidance(stateless=True)
def _gen_para(lm):
    lm += "<p>"
    # A few inline chunks is enough for a short demo page.
    lm += repeat(
        select(
            options=[
                _gen_text(),
                _gen_text_in_tag("em"),
                _gen_text_in_tag("strong"),
                "<br />",
            ],
        ),
        min=1,
        max=4,
    )
    lm += "</p>\n"
    return lm


@guidance(stateless=True)
def _gen_list_item(lm):
    lm += "<li>"
    lm += _gen_text()
    lm += "</li>\n"
    return lm


@guidance(stateless=True)
def _gen_ul(lm):
    lm += "<ul>\n"
    lm += repeat(_gen_list_item(), min=1, max=4)
    lm += "</ul>\n"
    return lm


@guidance(stateless=True)
def _gen_ol(lm):
    lm += "<ol>\n"
    lm += repeat(_gen_list_item(), min=1, max=4)
    lm += "</ol>\n"
    return lm


@guidance(stateless=True)
def _gen_list(lm):
    lm += select(options=[_gen_ul(), _gen_ol()])
    return lm


@guidance(stateless=True)
def _gen_body(lm):
    lm += "<body>\n"
    # Cap body blocks so a generation always finishes in reasonable time.
    lm += repeat(
        select(options=[_gen_heading(), _gen_para(), _gen_list()]),
        min=1,
        max=6,
    )
    lm += "</body>\n"
    return lm


@guidance(stateless=True)
def _gen_html(lm):
    lm += "<html>\n"
    lm += _gen_header()
    lm += _gen_body()
    lm += "</html>\n"
    return lm


@guidance(stateless=True)
def make_html(lm, name: str | None = None, *, temperature: float = 0.0):
    """Run the page grammar and stash the result under ``name`` on the LM."""
    return lm + capture(
        with_temperature(_gen_html(), temperature=temperature),
        name=name,
    )


def html_grammar():
    """Root grammar node — use ``.match(html)`` without calling the model."""
    # A10 imports this matcher for evaluation; it does not decode through it.
    return _gen_html()


# Keep this list in sync with what the grammar above can emit.
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
        self.root_count = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.stack:
            # A balanced second root is still invalid as a single HTML document.
            if tag != "html" or self.root_count:
                raise HtmlValidationError("Expected exactly one HTML document")
            self.root_count += 1
        self.saw_tag = True
        if tag not in ALLOWED_TAGS:
            raise HtmlValidationError(f"Disallowed tag: <{tag}>")
        # No href/onclick/style — attributes are how HTML gets dangerous fast.
        if attrs:
            raise HtmlValidationError(f"Attributes are not allowed on <{tag}>")
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self.stack:
            raise HtmlValidationError("Element outside HTML document")
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
        if not self.stack and data.strip():
            raise HtmlValidationError("Text outside HTML document")
        # Parser already decoded entities; bare angle brackets here are bad news.
        if "<" in data or ">" in data:
            raise HtmlValidationError("Raw '<' or '>' is not allowed in text nodes")

    def handle_comment(self, data: str) -> None:
        raise HtmlValidationError("Comments are outside the Assignment 9 grammar")

    def handle_decl(self, decl: str) -> None:
        raise HtmlValidationError("Declarations are outside the Assignment 9 grammar")

    def handle_pi(self, data: str) -> None:
        raise HtmlValidationError("Processing instructions are not allowed")


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


class _Elements(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes = []
        self.stack = []
        self.balanced = True
        self.allowed = True
        self.clean = True

    def handle_starttag(self, tag, attrs):
        self.allowed &= tag in ALLOWED_TAGS and not attrs
        node = {
            "tag": tag,
            "text": "",
            "parent": self.stack[-1] if self.stack else None,
        }
        self.nodes.append(node)
        if tag != "br":
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag != "br":
            self.balanced = False
            self.stack.pop()

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1]["tag"] == tag:
            self.stack.pop()
        else:
            self.balanced = False

    def handle_data(self, data):
        if not self.stack and data.strip():
            self.clean = False
        # Include inline emphasis text in the surrounding paragraph's visible text.
        for node in self.stack:
            node["text"] += data

    def handle_comment(self, data):
        self.clean = False

    def handle_decl(self, decl):
        self.clean = False

    def handle_pi(self, data):
        self.clean = False


def evaluate_html(document, requirements):
    """Return auditable checks and a 50/50 structure/content composite.

    Exact grammar is reported separately, never used for demo acceptance.
    Exact grammar compliance uses the returned string unchanged. DSPy trims field
    whitespace, so report a separate boundary-newline-tolerant match as diagnostic
    information only; it does not affect the score or exact pass rate.
    """
    parser = _Elements()
    parser.feed(document)
    parser.close()
    counts = Counter(node["tag"] for node in parser.nodes)
    relationships = {
        "html": {None},
        "head": {"html"},
        "body": {"html"},
        "title": {"head"},
        "h1": {"body"},
        "h2": {"body"},
        "p": {"body"},
        "ul": {"body"},
        "ol": {"body"},
        "li": {"ul", "ol"},
        "em": {"p"},
        "strong": {"p"},
        "br": {"p"},
    }
    shape = all(counts[tag] == 1 for tag in ("html", "head", "title", "body"))
    shape &= all(
        (node["parent"]["tag"] if node["parent"] else None)
        in relationships.get(node["tag"], set())
        for node in parser.nodes
    )
    shape &= [
        n["tag"] for n in parser.nodes if n["parent"] and n["parent"]["tag"] == "html"
    ] == ["head", "body"]
    # Matching is post-generation only; the original string is never repaired.
    grammar = html_grammar()
    exact = grammar.match(document) is not None
    structural = {
        "permitted_tags_and_attributes": bool(parser.nodes) and parser.allowed,
        "balanced_nesting": bool(parser.nodes) and parser.balanced and not parser.stack,
        "document_shape": bool(shape),
        "no_surrounding_content": bool(parser.nodes) and parser.clean,
    }
    content = {}
    for tag, expected in requirements["counts"].items():
        content[f"count:{tag}"] = counts[tag] == expected
    for tag, texts in requirements["texts"].items():
        actual = [" ".join(n["text"].split()) for n in parser.nodes if n["tag"] == tag]
        for text in texts:
            content[f"text:{tag}:{text}"] = bool(text.strip()) and text in actual
        if tag == "li":
            # Membership alone would incorrectly accept a reversed list.
            content["list_item_order"] = actual == texts
    # Weight structure and instructions equally even when requests have different
    # numbers of content checks. Exact serialization remains a separate diagnostic.
    structural_score = sum(structural.values()) / len(structural)
    instruction_score = sum(content.values()) / len(content)
    try:
        validate_html(document)
        whitelist_valid = True
    except HtmlValidationError:
        whitelist_valid = False
    return {
        "score": 0.5 * structural_score + 0.5 * instruction_score,
        "structural_score": structural_score,
        "instruction_score": instruction_score,
        "instruction_pass": all(content.values()),
        "grammar_pass": exact,
        "boundary_newline_match": grammar.match(document + "\n") is not None,
        "tag_nesting_valid": whitelist_valid and bool(shape),
        "structural_checks": structural,
        "instruction_checks": content,
    }


def metric(example, prediction, trace=None):
    """DSPy metric: partial credit for search, full score for demonstration acceptance."""
    return evaluate_html(prediction.html, example.requirements)["score"]
