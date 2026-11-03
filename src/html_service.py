"""
A small service that uses Outlines to constrain Qwen2.5-7B-Instruct's
output to valid HTML, via a context-free grammar, so the model generates
actual HTML tokens directly rather than filling in a template.
"""

from html.parser import HTMLParser

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Grammar is written in Lark syntax (not llama.cpp's GBNF format), since
# this is what Outlines' CFG type expects internally. Only <html>, <body>,
# <h1>, and <p> tags are allowed -- no list support. This is a deliberate
# simplification, but it has a real consequence: when the model wants to
# write a numbered list, it can't emit <ol>/<li>, so it works around the
# constraint by splitting each list item into two separate <p> tags (one
# for the number, one for the content). See PR Question 4 for examples.
HTML_GRAMMAR = outlines.types.CFG(r"""
?start: "<html><body>" heading paragraph+ "</body></html>"

heading: "<h1>" TEXT "</h1>"
paragraph: "<p>" TEXT "</p>"

TEXT: /[a-zA-Z0-9 .,!?']+/

%ignore " "
""")


def load_model(model_id=MODEL_ID):  # pragma: no cover
    """Load Qwen2.5-7B-Instruct in 4-bit, then wrap it with Outlines for constrained generation."""
    # Outlines works by restricting which tokens can be sampled at each
    # generation step, so it has to run in the same process as the model
    # weights. This is why I'm loading the model directly via transformers
    # rather than reusing my Assignment 7 llama.cpp HTTP server -- by the
    # time a response reaches my code from that server, every token has
    # already been chosen, which is too late for Outlines to constrain.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="cuda",
    )

    # outlines.from_transformers() requires the llguidance package to be
    # installed separately for CFG grammar support -- it's not pulled in
    # automatically by `uv add outlines`, and generation fails with a
    # ModuleNotFoundError at call time (not at import time) if it's missing.
    model = outlines.from_transformers(hf_model, tokenizer)
    return model


def generate_html(model, prompt, max_tokens=200):  # pragma: no cover
    """
    Generate HTML from the given prompt, constrained to HTML_GRAMMAR.

    Note: the prompt itself doesn't need to mention HTML at all -- the
    grammar is what forces the structured output, not the prompt wording.
    I verified this by sending a plain, unrelated question ("Tell me about
    the benefits of exercise") with no mention of HTML, and the grammar
    still produced valid, well-formed HTML output. This is the actual
    point of constrained decoding: the format is guaranteed by restricting
    which tokens can be sampled, not requested through clever prompting.
    """
    result = model(prompt, HTML_GRAMMAR, max_new_tokens=max_tokens)
    return result


ALLOWED_TAGS = {"html", "body", "h1", "p"}


class HTMLStructureValidator(HTMLParser):
    """
    Parses HTML and tracks whether every tag is one of ALLOWED_TAGS, and
    whether every opened tag is eventually closed in the correct order.
    """

    def __init__(self):
        super().__init__()
        self.tag_stack = []
        self.errors = []

    def handle_starttag(self, tag, attrs):
        if tag not in ALLOWED_TAGS:
            self.errors.append(f"Disallowed tag: <{tag}>")
        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        if not self.tag_stack:
            self.errors.append(f"Closing tag </{tag}> with no matching open tag")
            return
        expected = self.tag_stack.pop()
        if expected != tag:
            self.errors.append(f"Mismatched tag: expected </{expected}>, got </{tag}>")


def validate_html(html_string):
    """
    Validate that html_string is well-formed HTML using only allowed tags,
    with every tag properly opened and closed in order.
    Returns (is_valid, list_of_errors).

    Note: this checks structural correctness, not completeness -- an empty
    string has no tags and therefore no structural violations, so it
    correctly returns valid. This function is independent of the model and
    the Outlines grammar entirely, which is why it's the part of this
    module that's actually tested in CI (see tests/test_html.py);
    load_model() and generate_html() both require a GPU and a model
    download, so they're marked # pragma: no cover.
    """
    validator = HTMLStructureValidator()
    validator.feed(html_string)

    errors = list(validator.errors)
    if validator.tag_stack:
        errors.append(f"Unclosed tags remaining: {validator.tag_stack}")

    return len(errors) == 0, errors


if __name__ == "__main__":  # pragma: no cover
    model = load_model()
    print("Model loaded successfully with Outlines wrapper.")

    prompt = "Tell me about the benefits of exercise."
    print(f"\nPrompt: {prompt}")
    result = generate_html(model, prompt, max_tokens=400)
    print(f"\nGenerated HTML:\n{result}")
