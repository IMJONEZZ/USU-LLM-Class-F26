import pandas as pd

from src.preprocessing import (
    allocate_proportional_quotas,
    clean_resume,
    normalize_text,
    redact_pii,
    sample_stratified,
    strip_links_and_handles,
)

DIRTY_RESUME = (
    "   HR ADMINISTRATOR\xa0– Summary\n"
    "Email: jane.doe@example.com  Phone: (555) 123-4567\n"
    "See https://www.linkedin.com/in/janedoe or @janedoe\n"
    "• Managed “team” of 10 cafés\u200b  \n"
)


def test_normalize_text_maps_harmless_characters():
    assert normalize_text("a\xa0b–c\u200bd") == "a b-cd"
    assert normalize_text("café") == "cafe"


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


def make_resumes():
    rows = [
        {"ID": f"{category}{i}", "Category": category}
        for category, count in {"A": 10, "B": 10, "C": 2}.items()
        for i in range(count)
    ]
    return pd.DataFrame(rows)


def test_allocate_proportional_quotas_hand_checked():
    counts = pd.Series({"A": 10, "B": 10, "C": 2})
    assert allocate_proportional_quotas(counts, 11).to_dict() == {
        "A": 5,
        "B": 5,
        "C": 1,
    }


def test_allocate_proportional_quotas_sums_exactly_to_total():
    counts = pd.Series({"A": 10, "B": 10, "C": 2})
    quotas = allocate_proportional_quotas(counts, 10)
    assert quotas.sum() == 10
    assert quotas["C"] == 1


def test_sample_stratified_keeps_category_proportions():
    sampled = sample_stratified(make_resumes(), n_total=11, seed=0)
    assert sampled["Category"].value_counts().to_dict() == {"A": 5, "B": 5, "C": 1}


def test_sample_stratified_has_no_duplicate_resumes():
    sampled = sample_stratified(make_resumes(), n_total=11, seed=0)
    assert sampled["ID"].is_unique


def test_sample_stratified_is_reproducible_with_same_seed():
    first = sample_stratified(make_resumes(), n_total=11, seed=7)
    second = sample_stratified(make_resumes(), n_total=11, seed=7)
    pd.testing.assert_frame_equal(first, second)
