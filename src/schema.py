from html import escape
from html.parser import HTMLParser
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# The one list of tags both methods may produce. Method A's grammar imports CONTENT_TAGS too.
PAGE_TAGS = ["html", "head", "title", "body"]
CONTENT_TAGS = ["h1", "h2", "h3", "p", "em", "strong", "br"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Text(Strict):
    type: Literal["text"]
    text: str = Field(min_length=1)


class Em(Strict):
    type: Literal["em"]
    text: str = Field(min_length=1)


class Strong(Strict):
    type: Literal["strong"]
    text: str = Field(min_length=1)


class Br(Strict):
    type: Literal["br"]


Part = Annotated[Text | Em | Strong | Br, Field(discriminator="type")]


class Heading(Strict):
    type: Literal["heading"]
    level: Literal[1, 2, 3]
    text: str = Field(min_length=1)


class Paragraph(Strict):
    type: Literal["paragraph"]
    parts: list[Part] = Field(min_length=1)


Block = Annotated[Heading | Paragraph, Field(discriminator="type")]


class WebPage(Strict):
    title: str = Field(min_length=1)
    body: list[Block] = Field(min_length=1)


def part_to_html(part):
    match part:
        case Text():
            return escape(part.text, quote=False)
        case Em():
            return f"<em>{escape(part.text, quote=False)}</em>"
        case Strong():
            return f"<strong>{escape(part.text, quote=False)}</strong>"
        case Br():
            return "<br>"


def block_to_html(block):
    match block:
        case Heading():
            return f"<h{block.level}>{escape(block.text, quote=False)}</h{block.level}>"
        case Paragraph():
            return f"<p>{''.join(part_to_html(p) for p in block.parts)}</p>"


def to_html(page):
    body = "".join(block_to_html(b) for b in page.body)
    title = escape(page.title, quote=False)
    return f"<html><head><title>{title}</title></head><body>{body}</body></html>"


# What each tag may contain. This is the definition of "properly nested" for the validity check.
ALLOWED_CHILDREN = {
    "html": {"head", "body"},
    "head": {"title"},
    "title": set(),
    "body": {"h1", "h2", "h3", "p"},
    "h1": set(),
    "h2": set(),
    "h3": set(),
    "p": {"em", "strong", "br"},
    "em": set(),
    "strong": set(),
    "br": set(),
}
TEXT_PARENTS = {"title", "h1", "h2", "h3", "p", "em", "strong"}
VOID_TAGS = {"br"}
BLOCK_TAGS = {"h1", "h2", "h3", "p"}


class Checker(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.problems = [], []
        self.roots = self.titles = self.blocks = 0
        self.title_text = ""

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_CHILDREN:
            self.problems.append(f"tag not allowed: {tag}")
            return
        if attrs:
            self.problems.append(f"attributes not allowed on <{tag}>")
        self.check_parent(tag)
        self.titles += tag == "title"
        self.blocks += tag in BLOCK_TAGS
        if tag not in VOID_TAGS:
            self.stack.append(tag)

    def check_parent(self, tag):
        if not self.stack:
            self.roots += 1
            if tag != "html":
                self.problems.append(f"root must be <html>, got <{tag}>")
            elif self.roots > 1:
                self.problems.append("second root <html>")
        elif tag not in ALLOWED_CHILDREN[self.stack[-1]]:
            self.problems.append(f"<{tag}> not allowed inside <{self.stack[-1]}>")

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag not in ALLOWED_CHILDREN:
            self.problems.append(f"tag not allowed: {tag}")
        elif tag in VOID_TAGS:
            self.problems.append(f"end tag </{tag}> not allowed")
        elif not self.stack:
            self.problems.append(f"end tag </{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.problems.append(f"end tag </{tag}> does not match <{self.stack[-1]}>")
        else:
            self.stack.pop()

    def handle_data(self, data):
        if not data.strip():
            return
        parent = self.stack[-1] if self.stack else None
        if parent not in TEXT_PARENTS:
            where = f"<{parent}>" if parent else "outside any tag"
            self.problems.append(f"text not allowed in {where}")
        if parent == "title":
            self.title_text += data


def html_problems(html):
    """Every reason the page is invalid; an empty list means valid."""
    checker = Checker()
    checker.feed(html)
    checker.close()
    problems = checker.problems + [f"unclosed <{tag}>" for tag in checker.stack]
    if checker.titles != 1:
        problems.append("needs exactly one <title>")
    elif not checker.title_text.strip():
        problems.append("empty title")
    if checker.blocks == 0:
        problems.append("no heading or paragraph")
    return problems


def is_valid_html(html):
    return not html_problems(html)
