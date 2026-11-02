from pathlib import Path

from guidance import models

from src import config, guidance_html
from src.schema import CONTENT_TAGS, PAGE_TAGS, html_problems


def mock_page():
    """The tutorial grammar run on Guidance's Mock model (no GPU): random text, fixed structure."""
    lm = models.Mock(echo=False)
    lm += guidance_html.make_html(name="html_text")
    return lm["html_text"]


def test_a_page_from_the_tutorial_grammar_passes_the_shared_validity_check():
    assert html_problems(mock_page()) == []


def test_the_block_cap_keeps_the_page_finite():
    # without the cap the tutorial's one_or_more can loop; each body item is a heading or up to cap paragraphs
    assert mock_page().count("<p>") <= config.BLOCK_CAP * config.BLOCK_CAP


def test_grammar_source_mentions_every_tag_the_schema_allows():
    """Drift guard: schema.py's tag lists are the source of truth for both methods."""
    source = Path(guidance_html.__file__).read_text()
    missing = [
        t
        for t in PAGE_TAGS + CONTENT_TAGS
        if f"<{t}" not in source and f'"{t}"' not in source
    ]
    assert missing == []
