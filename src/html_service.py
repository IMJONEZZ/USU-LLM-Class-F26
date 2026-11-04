"""Generate HTML with an LLM, optionally forced to follow a grammar (Outlines).

The model is always passed in as an object with a ``generate`` method, so the
logic here (grammar, validator, evaluation) is tested with a fake model and CI
never needs torch or a model download. Only ``load_model`` imports the heavy
libraries, and it does so lazily.
"""

import argparse
import json
import re
import time
from html.parser import HTMLParser
from typing import Protocol

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 800

# Decision D4/D5: a small HTML subset. Text inside tags may use any character
# except "<", ">" and "&" (which could start a stray tag or character code) and
# control characters (tabs, line breaks, other invisible characters).
ALLOWED_TAGS = ("html", "head", "title", "body", "h1", "h2", "p", "ul", "li")

# Tags that never have a closing tag in ordinary HTML. Only the lenient check
# knows about them; the strict check allows none of them.
VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "hr", "img", "input", "link", "meta"}
)

HTML_GRAMMAR = r"""
start: "<html>" head body "</html>"
head: "<head><title>" TEXT "</title></head>"
body: "<body>" block+ "</body>"
block: h1 | h2 | p | ul
h1: "<h1>" TEXT "</h1>"
h2: "<h2>" TEXT "</h2>"
p: "<p>" TEXT "</p>"
ul: "<ul>" li+ "</ul>"
li: "<li>" TEXT "</li>"
TEXT: /[^<>&\x00-\x1f\x7f]+/
"""

SYSTEM_PROMPT = "You write short, valid HTML pages. Reply with only the HTML page."

# Ten fixed prompts for the constrained vs unconstrained comparison (D7).
EVAL_PROMPTS = (
    "Write a small HTML page about the planet Mars.",
    "Write a small HTML page about how bees make honey.",
    "Write a small HTML page listing three tips for studying.",
    "Write a small HTML page about the history of the bicycle.",
    "Write a small HTML page about the city of Logan, Utah.",
    "Write a small HTML page about why the sky is blue.",
    "Write a small HTML page with a recipe for pancakes.",
    "Write a small HTML page about how vaccines work.",
    "Write a small HTML page about the moon landing.",
    "Write a small HTML page about learning to play the guitar.",
)


class Generator(Protocol):
    """Anything that can turn a prompt into text, with or without a grammar."""

    def generate(
        self, prompt: str, grammar: str | None, max_new_tokens: int
    ) -> str: ...


class _TagChecker(HTMLParser):
    """Collects structural problems: unknown tags, bad nesting, stray text."""

    def __init__(self, strict: bool = True) -> None:
        super().__init__()
        self.strict = strict
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.first_tag: str | None = None

    def handle_starttag(self, tag, attrs):
        if self.first_tag is None:
            self.first_tag = tag
        if self.strict and tag not in ALLOWED_TAGS:
            self.errors.append(f"tag not allowed: <{tag}>")
        if not self.strict and tag in VOID_TAGS:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.strict and tag in VOID_TAGS:
            return
        if not self.stack:
            self.errors.append(f"closing </{tag}> with nothing open")
        elif self.stack[-1] != tag:
            self.errors.append(f"expected </{self.stack[-1]}> but found </{tag}>")
            if tag in self.stack:
                while self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()


def validate_html(text: str, strict: bool = True) -> list[str]:
    """Return a list of problems; an empty list means the page is valid.

    Strict (the check from my plan): starts with <html>, ends with </html>, uses
    only the allowed tags, and every tag is closed in the right order.

    Lenient (``strict=False``, a fairer extra check): also accepts a leading
    <!DOCTYPE html>, any tag name, and tags that never close (<meta>, <img>...).
    (Python's HTMLParser alone accepts anything, so the tag stack is what makes
    this a real check.)
    """
    errors: list[str] = []
    stripped = text.strip()
    if not strict:
        stripped = re.sub(r"^<!doctype html>\s*", "", stripped, flags=re.IGNORECASE)
    if not stripped:
        return ["empty output"]
    if not stripped.startswith("<html>" if strict else "<html"):
        errors.append("does not start with <html>")
    if not stripped.endswith("</html>"):
        errors.append("does not end with </html>")
    checker = _TagChecker(strict)
    checker.feed(stripped)
    checker.close()
    errors.extend(checker.errors)
    if checker.stack:
        errors.append("unclosed tags: " + ", ".join(checker.stack))
    return errors


def rescore_lenient(results: dict) -> dict:
    """Re-score saved eval pages with the lenient check (no model needed)."""
    out: dict = {
        "note": "lenient check; the strict result is the official one",
        "runs": {},
    }
    for mode, run in results["runs"].items():
        verdicts = [
            not validate_html(page["html"], strict=False) for page in run["pages"]
        ]
        valid = sum(verdicts)
        out["runs"][mode] = {
            "valid": valid,
            "total": len(verdicts),
            "valid_percent": round(100 * valid / len(verdicts), 1) if verdicts else 0.0,
            "per_page": verdicts,
        }
    return out


def generate_page(
    model: Generator,
    prompt: str,
    constrained: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Generate one page and report whether it is valid and how long it took."""
    start = time.perf_counter()
    html = model.generate(prompt, HTML_GRAMMAR if constrained else None, max_new_tokens)
    seconds = time.perf_counter() - start
    errors = validate_html(html)
    return {
        "prompt": prompt,
        "constrained": constrained,
        "html": html,
        "valid": not errors,
        "errors": errors,
        "seconds": round(seconds, 2),
    }


def evaluate(
    model: Generator,
    prompts: tuple[str, ...] = EVAL_PROMPTS,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict:
    """Same prompts, same model, with and without the grammar (decision D7)."""
    summary: dict = {"model": MODEL_NAME, "max_new_tokens": max_new_tokens, "runs": {}}
    for constrained in (True, False):
        pages = [generate_page(model, p, constrained, max_new_tokens) for p in prompts]
        valid = sum(page["valid"] for page in pages)
        summary["runs"]["constrained" if constrained else "unconstrained"] = {
            "valid": valid,
            "total": len(pages),
            "valid_percent": round(100 * valid / len(pages), 1) if pages else 0.0,
            "mean_seconds": round(sum(p["seconds"] for p in pages) / len(pages), 2)
            if pages
            else 0.0,
            "pages": pages,
        }
    return summary


class OutlinesModel:
    """Wraps an Outlines model plus its tokenizer behind ``Generator``."""

    def __init__(self, model, tokenizer, cfg_type):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg_type = cfg_type

    def generate(self, prompt: str, grammar: str | None, max_new_tokens: int) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if grammar is None:
            return self.model(text, max_new_tokens=max_new_tokens)
        return self.model(text, self.cfg_type(grammar), max_new_tokens=max_new_tokens)


def load_model(name: str = MODEL_NAME) -> OutlinesModel:  # pragma: no cover
    """Load the real model on the Apple GPU (MPS). Imports are lazy on purpose.

    Not covered by tests: it needs torch, a ~3 GB download and a GPU, so CI
    cannot run it. Everything it feeds (``OutlinesModel``) is tested with fakes.
    """
    import outlines
    import torch
    from outlines.types import CFG
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(name)
    hf_model = AutoModelForCausalLM.from_pretrained(name, dtype="auto").to(device)
    return OutlinesModel(
        outlines.from_transformers(hf_model, tokenizer), tokenizer, CFG
    )


def main(argv: list[str] | None = None) -> int:
    """Command line: ``python -m src.html_service eval --out results.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="constrained vs unconstrained on 10 prompts")
    ev.add_argument("--out", default="assignment_9/results/eval_results.json")
    rs = sub.add_parser("rescore", help="re-score saved eval pages leniently")
    rs.add_argument("--results", default="assignment_9/results/eval_results.json")
    rs.add_argument("--out", default="assignment_9/results/lenient_rescore.json")
    args = parser.parse_args(argv)
    if args.command == "rescore":
        with open(args.results) as handle:
            rescored = rescore_lenient(json.load(handle))
        with open(args.out, "w") as handle:
            json.dump(rescored, handle, indent=2)
        for mode, run in rescored["runs"].items():
            print(
                f"lenient {mode}: {run['valid']}/{run['total']} valid ({run['valid_percent']}%)"
            )
        return 0
    if args.command == "eval":
        result = evaluate(load_model())
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)
        for mode, run in result["runs"].items():
            print(
                f"{mode}: {run['valid']}/{run['total']} valid "
                f"({run['valid_percent']}%), mean {run['mean_seconds']} s"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
