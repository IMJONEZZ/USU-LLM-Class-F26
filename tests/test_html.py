import typing
from html.parser import HTMLParser

import pytest

from src import schema
from src.html import ALLOWED_CHILDREN, html_problems, to_html
from src.schema import Br, Em, Heading, Paragraph, Strong, Text, WebPage


def members(annotated):
    """The classes inside Annotated[A | B | ..., Field(discriminator=...)]."""
    return typing.get_args(typing.get_args(annotated)[0])


def exhaustive_page():
    """One page that uses every variant the schema allows (no two Text parts next to each other)."""
    return WebPage(
        title="All variants",
        body=[
            Heading(type="heading", level=1, text="one"),
            Heading(type="heading", level=2, text="two"),
            Heading(type="heading", level=3, text="three"),
            Paragraph(
                type="paragraph",
                parts=[
                    Text(type="text", text="plain "),
                    Em(type="em", text="emphasized"),
                    Text(type="text", text=" then "),
                    Strong(type="strong", text="strong"),
                    Br(type="br"),
                    Text(type="text", text="after the break"),
                ],
            ),
        ],
    )


class Parser(HTMLParser):
    """Independent parse-back: HTML -> WebPage. Lives in the test so the assembler is not checked by itself."""

    def __init__(self):
        super().__init__()
        self.stack, self.title, self.body, self.parts, self.heading = [], "", [], [], ""
        self.tags = set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        if tag == "br":
            self.parts.append(Br(type="br"))
        else:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag
        if tag == "p":
            self.body.append(Paragraph(type="paragraph", parts=self.parts))
            self.parts = []
        if tag in ("h1", "h2", "h3"):
            self.body.append(
                Heading(type="heading", level=int(tag[1]), text=self.heading)
            )

    def handle_data(self, data):
        where = self.stack[-1]
        if where == "title":
            self.title = data
        elif where in ("h1", "h2", "h3"):
            self.heading = data
        elif where == "em":
            self.parts.append(Em(type="em", text=data))
        elif where == "strong":
            self.parts.append(Strong(type="strong", text=data))
        elif where == "p":
            self.parts.append(Text(type="text", text=data))


def parse_back(html):
    parser = Parser()
    parser.feed(html)
    assert parser.stack == []
    return WebPage(title=parser.title, body=parser.body), parser.tags


# ---- every schema variant has an assembler branch ----

PART_SAMPLES = {
    Text: (Text(type="text", text="hi"), "hi"),
    Em: (Em(type="em", text="hi"), "<em>hi</em>"),
    Strong: (Strong(type="strong", text="hi"), "<strong>hi</strong>"),
    Br: (Br(type="br"), "<br>"),
}


def test_every_part_variant_has_a_sample_and_an_assembler_branch():
    assert set(members(schema.Part)) == set(PART_SAMPLES)
    for part, expected in PART_SAMPLES.values():
        html = to_html(
            WebPage(title="t", body=[Paragraph(type="paragraph", parts=[part])])
        )
        assert f"<p>{expected}</p>" in html


def test_every_block_variant_is_covered_by_these_tests():
    assert set(members(schema.Block)) == {Heading, Paragraph}


@pytest.mark.parametrize(
    "level", typing.get_args(Heading.model_fields["level"].annotation)
)
def test_every_heading_level_becomes_its_own_tag(level):
    html = to_html(
        WebPage(title="t", body=[Heading(type="heading", level=level, text="x")])
    )
    assert f"<h{level}>x</h{level}>" in html


# ---- assembler output matches the schema, both directions ----


def test_tags_in_assembled_html_equal_the_allowed_tags_exactly():
    _, tags = parse_back(to_html(exhaustive_page()))
    assert tags == set(ALLOWED_CHILDREN)


def test_round_trip_html_back_to_the_same_page():
    page = exhaustive_page()
    parsed, _ = parse_back(to_html(page))
    assert parsed == page


# ---- hand-checked cases ----


def test_hand_built_page_gives_exact_html():
    page = WebPage(
        title="Cats",
        body=[
            Heading(type="heading", level=1, text="Cats"),
            Paragraph(type="paragraph", parts=[Text(type="text", text="Meow.")]),
        ],
    )
    assert to_html(page) == (
        "<html><head><title>Cats</title></head>"
        "<body><h1>Cats</h1><p>Meow.</p></body></html>"
    )


def test_text_is_escaped_so_it_cannot_inject_tags():
    page = WebPage(
        title="a < b",
        body=[Paragraph(type="paragraph", parts=[Text(type="text", text="x & <y>")])],
    )
    html = to_html(page)
    assert "<title>a &lt; b</title>" in html
    assert "<p>x &amp; &lt;y&gt;</p>" in html


# ---- shared validity check ----

BODY = "<html><head><title>t</title></head><body>{}</body></html>"

BAD_HTML = {
    "unclosed tag": (
        BODY.format("<p>hi").replace("</body></html>", ""),
        "unclosed <p>",
    ),
    "stray end tag": (BODY.format("<p>x</p>") + "</p>", "nothing open"),
    "mismatched end": (BODY.format("<p><em>x</p></em>"), "does not match"),
    "tag not allowed": (BODY.format("<div>x</div>"), "tag not allowed: div"),
    "block inside paragraph": (
        BODY.format("<p><h1>x</h1></p>"),
        "<h1> not allowed inside <p>",
    ),
    "paragraph inside paragraph": (
        BODY.format("<p>a<p>b</p></p>"),
        "<p> not allowed inside <p>",
    ),
    "missing title": (
        "<html><head></head><body><p>x</p></body></html>",
        "exactly one <title>",
    ),
    "two titles": (
        "<html><head><title>a</title><title>b</title></head><body><p>x</p></body></html>",
        "exactly one <title>",
    ),
    "empty title": (
        "<html><head><title> </title></head><body><p>x</p></body></html>",
        "empty title",
    ),
    "empty body": (BODY.format(""), "no heading or paragraph"),
    "prose before the page": (
        "Sure! Here you go: " + BODY.format("<p>x</p>"),
        "text not allowed",
    ),
    "text directly in body": (BODY.format("hi<p>x</p>"), "text not allowed in <body>"),
    "end tag for br": (BODY.format("<p>a<br></br></p>"), "end tag </br>"),
    "no html root": ("<p>hi</p>", "root must be <html>"),
    "attribute": (BODY.format('<p class="x">x</p>'), "attributes not allowed"),
    "two roots": (BODY.format("<p>x</p>") * 2, "second root"),
    "empty string": ("", "no heading or paragraph"),
}


def test_the_assembled_exhaustive_page_is_valid():
    assert html_problems(to_html(exhaustive_page())) == []


@pytest.mark.parametrize("name", BAD_HTML)
def test_each_broken_page_is_rejected_for_the_right_reason(name):
    html, reason = BAD_HTML[name]
    assert html_problems(html)
    assert any(reason in problem for problem in html_problems(html)), html_problems(
        html
    )


def test_self_closing_br_is_valid():
    assert not html_problems(BODY.format("<p>a<br/>b</p>"))


def test_escaped_characters_are_valid_text():
    assert not html_problems(BODY.format("<p>a &lt; b &amp; c</p>"))


@pytest.mark.parametrize("part", [sample for sample, _ in PART_SAMPLES.values()])
def test_every_single_variant_page_is_valid(part):
    assert not html_problems(
        to_html(WebPage(title="t", body=[Paragraph(type="paragraph", parts=[part])]))
    )
