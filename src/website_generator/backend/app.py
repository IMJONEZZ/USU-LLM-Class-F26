"""Local grammar-constrained HTML generator. Run from the project directory."""

import os
import re
from contextlib import asynccontextmanager
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent.parent
TAGS = {
    "div",
    "section",
    "article",
    "h1",
    "h2",
    "h3",
    "p",
    "strong",
    "em",
    "ul",
    "li",
    "a",
}
INLINE = {"strong", "em", "a"}
ENTITIES = {"amp", "lt", "gt", "quot", "#39"}
TEXT = re.compile(r"[^<>&\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+|&(amp|lt|gt|quot|#39);")
URL = re.compile(
    r"https://[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9/_.~-]*)?"
)


class SafeHTML(HTMLParser):
    """Reject outside the grammar, then serialize a safe canonical fragment."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack = []
        self.parts = []
        self.elements = 0
        self.items = []

    def handle_starttag(self, tag, attrs):
        parent = self.stack[-1] if self.stack else None
        if tag not in TAGS or len(self.stack) >= 64:
            raise ValueError("Unsupported element or nesting depth")
        if parent is None and tag in INLINE | {"li"}:
            raise ValueError("Invalid root element")
        if parent == "ul" and tag != "li":
            raise ValueError("Lists require list items")
        if tag == "li" and parent != "ul":
            raise ValueError("List item outside list")
        if parent in {"h1", "h2", "h3", "p", "strong", "em"} and tag not in INLINE:
            raise ValueError("Invalid inline content")
        if parent == "a" or (
            parent in {"div", "section", "article", "li"} and tag in INLINE
        ):
            raise ValueError("Invalid content")
        if tag == "a":
            if len(attrs) != 1 or attrs[0][0] != "href":
                raise ValueError("Only href is allowed")
            url = attrs[0][1] or ""
            parsed = urlsplit(url)
            if (
                not URL.fullmatch(url)
                or not parsed.hostname
                or any(
                    not label or label.startswith("-") or label.endswith("-")
                    for label in parsed.hostname.split(".")
                )
            ):
                raise ValueError("Invalid HTTPS URL")
            opening = f'<a href="{escape(url, quote=True)}">'
        else:
            if attrs:
                raise ValueError("Attributes are forbidden")
            opening = f"<{tag}>"
        if self.get_starttag_text() != opening:
            raise ValueError("Noncanonical markup")
        if tag == "li":
            self.items[-1] += 1
        if tag == "ul":
            self.items.append(0)
        self.stack.append(tag)
        self.parts.append(opening)
        self.elements += 1

    def handle_endtag(self, tag):
        if tag == "a" and self.parts and self.parts[-1].startswith('<a href="'):
            raise ValueError("Links require text")
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("Unbalanced HTML")
        if tag == "ul" and self.items.pop() == 0:
            raise ValueError("Empty list")
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        if (not self.stack or self.stack[-1] == "ul") and data.strip(" \t\r\n"):
            raise ValueError("Unexpected text")
        if "".join(m.group() for m in TEXT.finditer(data)) != data:
            raise ValueError("Invalid text")
        self.parts.append(escape(data, quote=False))

    def handle_entityref(self, name):
        if name not in ENTITIES or not self.stack or self.stack[-1] == "ul":
            raise ValueError("Invalid entity")
        self.parts.append(f"&{name};")

    def handle_charref(self, name):
        self.handle_entityref("#" + name)

    def handle_comment(self, data):
        raise ValueError("Comments are forbidden")

    def handle_decl(self, decl):
        raise ValueError("Declarations are forbidden")

    def handle_pi(self, data):
        raise ValueError("Processing instructions are forbidden")

    def unknown_decl(self, data):
        raise ValueError("Unknown declaration")


def sanitize_html(source: str) -> str:
    if not source or len(source) > 100_000:
        raise ValueError("Empty or oversized HTML")
    parser = SafeHTML()
    parser.feed(source)
    parser.close()
    result = "".join(parser.parts)
    # Also catches HTMLParser's recovery of malformed closing tags/entities.
    if parser.stack or not parser.elements or result != source:
        raise ValueError("Incomplete or noncanonical HTML")
    return result


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=4000)

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Prompt must contain text")
        return value.strip()


@asynccontextmanager
async def lifespan(app):
    app.state.grammar = (ROOT / "backend/grammar.gbnf").read_text()
    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        app.state.client = client
        yield


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: GenerateRequest):
    payload = {
        "model": os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        "messages": [
            {
                "role": "system",
                "content": "Create a short HTML fragment for the user. "
                "Use div, section, article, h1-h3, p, strong, em, ul, li and HTTPS links. "
                "No CSS, attributes except link href, markdown, or explanations. "
                "Put strong, em and links inside paragraphs or headings. Keep it concise.",
            },
            {"role": "user", "content": request.prompt},
        ],
        "max_tokens": 1600,
        "temperature": 0.6,
        "structured_outputs": {"grammar": app.state.grammar},
    }
    try:
        response = await app.state.client.post(
            "http://127.0.0.1:8000/v1/chat/completions", json=payload
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice["finish_reason"] != "stop":
            raise HTTPException(
                502, "Generation incomplete. Try a shorter website description."
            )
        html = choice["message"]["content"]
        if not isinstance(html, str):
            raise TypeError("Missing HTML")
        return {"html": sanitize_html(html)}
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "Generation timed out. Try again.") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            502, "Cannot generate HTML. Check the local vLLM server."
        ) from exc
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise HTTPException(
            502, "The model returned invalid HTML or an invalid response."
        ) from exc


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend/index.html")


app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
