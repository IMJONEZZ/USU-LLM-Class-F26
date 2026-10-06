import pandas as pd

from src.preprocessing import (
    allocate_proportional_quotas,
    clean_resume,
    clean_resume_column,
    make_splits,
    normalize_text,
    redact_pii,
    sample_stratified,
    split_stratified,
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


def test_clean_resume_column_cleans_the_text_and_keeps_other_columns():
    resumes = pd.DataFrame(
        {
            "ID": [1, 2],
            "Resume_str": ["  Chef\xa0jane@example.com  ", "Cook (555) 123-4567"],
            "Category": ["CHEF", "CHEF"],
        }
    )

    cleaned_resumes = clean_resume_column(resumes)

    assert list(cleaned_resumes["Resume_str"]) == ["Chef [EMAIL]", "Cook [PHONE]"]
    assert list(cleaned_resumes["ID"]) == [1, 2]
    assert list(cleaned_resumes["Category"]) == ["CHEF", "CHEF"]


def test_clean_resume_column_does_not_modify_the_input_frame():
    resumes = pd.DataFrame({"Resume_str": ["  Chef  "]})

    clean_resume_column(resumes)

    assert list(resumes["Resume_str"]) == ["  Chef  "]


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


def make_resumes_for_split():
    rows = [
        {"ID": f"{category}{i}", "Category": category}
        for category, count in {"A": 20, "B": 10, "C": 3}.items()
        for i in range(count)
    ]
    return pd.DataFrame(rows)


def split_counts(split_resumes):
    return split_resumes["Category"].value_counts().to_dict()


def test_split_stratified_sets_have_no_id_overlap():
    train, validation, test = split_stratified(make_resumes_for_split(), seed=0)
    train_ids, validation_ids, test_ids = (
        set(train["ID"]),
        set(validation["ID"]),
        set(test["ID"]),
    )
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_split_stratified_covers_every_resume_exactly_once():
    resumes = make_resumes_for_split()
    train, validation, test = split_stratified(resumes, seed=0)
    all_ids = list(train["ID"]) + list(validation["ID"]) + list(test["ID"])
    assert sorted(all_ids) == sorted(resumes["ID"])


def test_split_stratified_keeps_70_20_10_within_each_category():
    train, validation, test = split_stratified(make_resumes_for_split(), seed=0)
    assert split_counts(train) == {"A": 14, "B": 7, "C": 2}
    assert split_counts(validation) == {"A": 4, "B": 2, "C": 1}
    assert split_counts(test) == {"A": 2, "B": 1}


def test_split_stratified_handles_category_smaller_than_ten_rows():
    train, validation, test = split_stratified(make_resumes_for_split(), seed=0)
    sparse_counts = [
        split_counts(split).get("C", 0) for split in (train, validation, test)
    ]
    assert sparse_counts == [2, 1, 0]


def test_split_stratified_is_reproducible_with_same_seed():
    first = split_stratified(make_resumes_for_split(), seed=7)
    second = split_stratified(make_resumes_for_split(), seed=7)
    for first_split, second_split in zip(first, second):
        pd.testing.assert_frame_equal(first_split, second_split)


def test_split_stratified_stratifies_by_the_column_it_is_given():
    resumes = make_resumes_for_split().rename(columns={"Category": "Group"})
    train, _, _ = split_stratified(resumes, stratify_by="Group", seed=0)
    assert train["Group"].value_counts().to_dict() == {"A": 14, "B": 7, "C": 2}


def test_make_splits_cleans_the_text_then_samples_and_splits():
    resumes = pd.DataFrame(
        {
            "ID": range(30),
            "Resume_str": ["  Chef\xa0jane@example.com  "] * 30,
            "Category": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        }
    )

    train, validation, test = make_splits(resumes)

    assert (len(train), len(validation), len(test)) == (21, 6, 3)
    assert set(train["Resume_str"]) == {"Chef [EMAIL]"}
