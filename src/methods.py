import json

from guidance import models
from pydantic import ValidationError

from src import config, guidance_html
from src.schema import WebPage, html_problems, to_html


def run_guidance(model, tokenizer, chat):
    """Method A: the tutorial's make_html grammar. `chat` is the chat-formatted prompt text, the same
    string Method B starts from. Returns the generated HTML and whether the token cap stopped it
    before the page was complete."""
    lm = models.Transformers(model, tokenizer, echo=False)
    generated = ""
    page = guidance_html.make_html(name="html_text", temperature=config.TEMPERATURE)
    for state in lm.stream() + (chat + page):
        generated = str(state)[len(chat) :]
        if (
            len(tokenizer.encode(generated, add_special_tokens=False))
            >= config.MAX_NEW_TOKENS
        ):
            break
    new_tokens = len(tokenizer.encode(generated, add_special_tokens=False))
    # a page that finishes exactly at the cap is complete, not truncated
    return {
        "text": generated,
        "new_tokens": new_tokens,
        "hit_cap": new_tokens >= config.MAX_NEW_TOKENS
        and not generated.rstrip().endswith("</html>"),
    }


def build_prompt(request):
    """Method B's prompt: exactly Method A's prompt, plus the schema text."""
    schema_text = json.dumps(WebPage.model_json_schema())
    base = config.BASE_PROMPT_TEMPLATE.format(request=request)
    return f"{base}\n\n{config.SCHEMA_INSTRUCTION}\n{schema_text}"


def bucket_for(problems, hit_cap):
    """No problems means valid, even if the cap was reached: a complete page is a complete page."""
    if not problems:
        return "valid", ""
    return ("truncated" if hit_cap else "malformed"), problems[0]


def classify_json(text, hit_cap):
    """Method B: strict, the raw model text must itself be the JSON (no prose, no markdown fence)."""
    try:
        page = WebPage.model_validate_json(text)
    except ValidationError as error:
        return bucket_for([error.errors()[0]["msg"]], hit_cap)
    return bucket_for(html_problems(to_html(page)), hit_cap)


def json_to_html(text):
    """The HTML Method B's JSON assembles to, or None when the JSON does not validate."""
    try:
        return to_html(WebPage.model_validate_json(text))
    except ValidationError:
        return None


def classify_html(text, hit_cap):
    """Method A: the grammar writes HTML directly, so the shared validity check judges it."""
    return bucket_for(html_problems(text), hit_cap)
