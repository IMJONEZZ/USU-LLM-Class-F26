import pytest

from src.preprocessing import (
    clean_resume,
    garbage_ratio,
    normalize_text,
    redact_pii,
    strip_links_and_handles,
    whitespace_ratio,
)

DIRTY_RESUME = (
    "   HR ADMINISTRATOR\xa0\u2013 Summary\n"
    "Email: jane.doe@example.com  Phone: (555) 123-4567\n"
    "See https://www.linkedin.com/in/janedoe or @janedoe\n"
    "\u2022 Managed \u201cteam\u201d of 10 caf\u00e9s\u200b  \n"
)


def test_normalize_text_maps_harmless_characters():
    assert normalize_text("a\xa0b\u2013c\u200bd") == "a b-cd"
    assert normalize_text("caf\u00e9") == "cafe"


def test_strip_links_and_handles_removes_url_and_handle():
    result = strip_links_and_handles("see https://x.com/a and @janedoe now")
    assert "x.com" not in result
    assert "@janedoe" not in result


def test_strip_links_and_handles_redacts_email_before_handle_rule():
    assert strip_links_and_handles("mail a.b@c.com") == "mail [EMAIL]"


def test_redact_pii_replaces_phone():
    assert redact_pii("call (555) 123-4567 today") == "call [PHONE] today"


def test_clean_resume_removes_all_pii_and_links():
    result = clean_resume(DIRTY_RESUME)
    assert "example.com" not in result
    assert "555" not in result
    assert "linkedin" not in result
    assert "@" not in result
    assert result.startswith("HR ADMINISTRATOR")
    assert "cafes" in result


def test_garbage_ratio_is_zero_for_harmless_non_ascii():
    assert garbage_ratio("a\xa0b \u2013 \u2022 caf\u00e9") == 0.0


def test_garbage_ratio_counts_replacement_character():
    assert garbage_ratio("abc\ufffd") == pytest.approx(0.25)


def test_garbage_ratio_flags_mojibake():
    assert garbage_ratio("it\u00e2\u20ac\u2122s") > 0


def test_whitespace_ratio_hand_checked():
    assert whitespace_ratio("a b") == pytest.approx(1 / 3)


@pytest.mark.parametrize("empty_text", ["", "   ", "\n\t"])
def test_ratios_do_not_crash_on_empty_or_whitespace(empty_text):
    assert garbage_ratio(empty_text) == 0.0
    assert whitespace_ratio(empty_text) in (0.0, 1.0)
