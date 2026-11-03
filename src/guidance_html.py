# The Guidance tutorial's HTML grammar (https://github.com/guidance-ai/guidance), as given.
# Deviation from the tutorial: capped_one_or_more replaces the three one_or_more calls, because the
# Action Plan caps that loop so a run cannot spend its tokens repeating blocks forever.
from guidance import capture, gen, guidance, optional, select, with_temperature
from guidance.models import Model

from src import config


def capped_one_or_more(item, times=config.BLOCK_CAP):
    """item, optionally followed by more items, at most `times` in total."""
    grammar = item
    for _ in range(times - 1):
        grammar = item + optional(grammar)
    return grammar


@guidance(stateless=True)
def _gen_text(lm: Model):
    return lm + gen(regex="[^<>]+")


@guidance(stateless=True)
def _gen_text_in_tag(lm: Model, tag: str):
    lm += f"<{tag}>"
    lm += _gen_text()
    lm += f"</{tag}>"
    return lm


@guidance(stateless=True)
def _gen_header(lm: Model):
    lm += "<head>\n"
    lm += _gen_text_in_tag("title") + "\n"
    lm += "</head>\n"
    return lm


@guidance(stateless=True)
def _gen_heading(lm: Model):
    lm += select(
        options=[_gen_text_in_tag("h1"), _gen_text_in_tag("h2"), _gen_text_in_tag("h3")]
    )
    lm += "\n"
    return lm


@guidance(stateless=True)
def _gen_para(lm: Model):
    lm += "<p>"
    lm += capped_one_or_more(
        select(
            options=[
                _gen_text(),
                _gen_text_in_tag("em"),
                _gen_text_in_tag("strong"),
                "<br />",
            ],
        )
    )
    lm += "</p>\n"
    return lm


@guidance(stateless=True)
def _gen_body(lm: Model):
    lm += "<body>\n"
    lm += capped_one_or_more(
        select(options=[_gen_heading(), capped_one_or_more(_gen_para())])
    )
    lm += "</body>\n"
    return lm


@guidance(stateless=True)
def _gen_html(lm: Model):
    lm += "<html>\n"
    lm += _gen_header()
    lm += _gen_body()
    lm += "</html>\n"
    return lm


@guidance(stateless=True)
def make_html(
    lm,
    name: str | None = None,
    *,
    temperature: float = 0.0,
):
    return lm + capture(
        with_temperature(_gen_html(), temperature=temperature),
        name=name,
    )
