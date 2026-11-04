"""Guidance compositional grammar for Assignment 9 HTML pages.

Tags and nesting are forced during decoding (llguidance). The model only picks
among allowed branches and fills plain text. ``html_grammar().match(...)`` is
the offline "does this string still fit the grammar?" check.
"""

from __future__ import annotations

from guidance import gen, guidance, select
from guidance.library import capture, with_temperature

# repeat isn't re-exported from guidance.library.__init__, but this is the
# same helper one_or_more uses under the hood — we need max= so Mock/Qwen
# can't spin forever on one_or_more.
from guidance.library._sequences import repeat


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
    return _gen_html()
