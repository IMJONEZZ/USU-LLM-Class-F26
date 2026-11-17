"""Client for the HTML service, plus the check that the HTML is really HTML.

The server in `gpu/serve_html_app.py` runs llama.cpp with `--grammar-file`, so
the constraint lives next to the weights and every response is already markup
by the time it reaches here. This module asks for a page and then verifies it,
which sounds redundant and isn't: the grammar is a claim about the server I'm
running, and if I ever point this at a server without the grammar loaded I'd
rather find out from a failing check than from a page full of apologies.

Not named `src/html.py`. That would sit next to a stdlib module called `html`,
and while Python 3 resolves it fine I didn't want to spend an evening finding
that out the hard way.

    export LLAMA_API_KEY=...
    uv run python -m src.html_output "a page about Tatooine"
"""

import pathlib
import re
from html.parser import HTMLParser

from src.inference import ask

BASE_URL = "https://froostysnoowman--hw9-serve-html-serve.us-east.modal.run"
MODEL = "qwen2.5-7b-html"

# A page is a few hundred tokens and the grammar lets it run to ten blocks, so
# the assignment 7 default of 200 would cut one off halfway through.
MAX_TOKENS = 900

# Assignment 7 waits 6 times at 5 seconds, which was plenty for an 800MB model
# on CPU. This one is 4.7GB going onto a GPU, and a cold container spent well
# over a minute loading it, so the old budget would have given up while the
# server was still doing exactly what it was supposed to.
COLD_START_ATTEMPTS = 20
COLD_START_SECONDS = 10.0

# These have to agree with gpu/html.gbnf. They're written out here rather than
# parsed out of the grammar at import time, because a validator that reads its
# rules from the same file it's checking against can't catch the two drifting
# apart. There's a test that compares them instead.
ALLOWED_TAGS = ("html", "head", "title", "body", "h1", "h2", "p", "ul", "li")

GRAMMAR_FILE = pathlib.Path(__file__).resolve().parents[1] / "gpu" / "html.gbnf"


def grammar_tags(path=None):
    """Pull the tag names out of the grammar file.

    Only used by the tests, to prove the grammar and `ALLOWED_TAGS` still
    describe the same set of tags.
    """
    text = pathlib.Path(path or GRAMMAR_FILE).read_text(encoding="utf-8")
    body = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )
    return {m.lower() for m in re.findall(r"<([a-zA-Z][a-zA-Z0-9]*)>", body)}


class _Checker(HTMLParser):
    """Walks the markup and records everything wrong with it.

    HTMLParser is forgiving by design, so it won't complain about a tag that
    never closes. Keeping the open tags on a stack is what turns it into
    something that will.
    """

    def __init__(self, allowed=ALLOWED_TAGS):
        super().__init__(convert_charrefs=True)
        self.allowed = tuple(allowed)
        self.open_tags = []
        self.problems = []
        self.saw_a_tag = False

    def handle_starttag(self, tag, attrs):
        self.saw_a_tag = True
        if tag not in self.allowed:
            self.problems.append(f"<{tag}> is not a tag the grammar allows")
        if attrs:
            self.problems.append(f"<{tag}> carries attributes and the grammar has none")
        self.open_tags.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.saw_a_tag = True
        self.problems.append(
            f"<{tag}/> is self closing and the grammar has no such tag"
        )

    def handle_endtag(self, tag):
        self.saw_a_tag = True
        if not self.open_tags:
            self.problems.append(f"</{tag}> closes something that was never opened")
        elif self.open_tags[-1] != tag:
            self.problems.append(f"</{tag}> closes <{self.open_tags[-1]}>")
        else:
            self.open_tags.pop()


def validate(markup, allowed=ALLOWED_TAGS):
    """Return a list of everything wrong with this markup. Empty means good."""
    checker = _Checker(allowed=allowed)
    checker.feed(markup)
    checker.close()
    problems = list(checker.problems)
    problems += [f"<{tag}> is never closed" for tag in reversed(checker.open_tags)]
    # Markup with no tags in it parses perfectly happily, so without this an
    # empty body or a paragraph of plain English comes back clean. That's the
    # exact shape of the failure I'm worried about, which is the server coming
    # up without the grammar loaded and answering in prose.
    if not checker.saw_a_tag:
        problems.append("no HTML tags at all, this is plain text")
    return problems


def is_valid(markup, allowed=ALLOWED_TAGS):
    return not validate(markup, allowed=allowed)


def generate(
    topic,
    url=BASE_URL,
    model=MODEL,
    max_tokens=MAX_TOKENS,
    session=None,
    attempts=COLD_START_ATTEMPTS,
    retry_seconds=COLD_START_SECONDS,
    **kwargs,
):
    """Ask the service for a page and hand back the markup with its problems.

    This goes to the ordinary chat completions endpoint and sends no grammar
    of its own. I had this down as the hard part, because llama.cpp documents
    `grammar` as a field on `/completion` and the chat endpoint only takes a
    JSON schema, so I expected to give up the chat template and format the
    prompt by hand. It turns out `--grammar-file` is a server-wide default
    sampler and applies to both endpoints, so the chat route is constrained
    too and the template still gets applied.

    The problems come back alongside the markup instead of raising. A page
    that fails the check is the single most interesting thing this service can
    produce, and throwing it away to raise an exception would lose it.
    """
    markup = ask(
        topic,
        url=url,
        model=model,
        max_tokens=max_tokens,
        session=session,
        attempts=attempts,
        retry_seconds=retry_seconds,
        **kwargs,
    )
    return {"topic": topic, "html": markup, "problems": validate(markup)}


def show(topic, **kwargs):
    """Print the request and the response, which is what the PR screenshot is."""
    import json
    import time

    url = kwargs.get("url", BASE_URL)
    payload = {
        "model": kwargs.get("model", MODEL),
        "messages": [{"role": "user", "content": topic}],
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
        "temperature": 0.0,
    }
    print("REQUEST")
    print(f"  POST {url}/v1/chat/completions")
    print("  Authorization: Bearer ****")
    print("  Content-Type: application/json")
    for line in json.dumps(payload, indent=2).splitlines():
        print(f"  {line}")
    print("\n  no grammar in this body. the server already has one.")

    started = time.time()
    result = generate(topic, **kwargs)
    print(f"\nRESPONSE  ({time.time() - started:.1f}s)")
    for line in result["html"].splitlines():
        print(f"  {line}")

    problems = result["problems"]
    print("\nCHECK")
    if problems:
        for problem in problems:
            print(f"  {problem}")
    else:
        tags = len(re.findall(r"</([a-zA-Z0-9]+)>", result["html"]))
        print(f"  valid: {tags} closed tags, nothing outside the grammar")
    return result


if __name__ == "__main__":  # pragma: no cover - entry point, not logic
    import sys

    asked = sys.argv[1] if len(sys.argv) > 1 else "a short page about Tatooine"
    show(asked)
