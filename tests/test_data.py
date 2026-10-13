import json

import pytest

from src import config, data, teacher
from tests.helpers import (
    LONG_Q,
    QUESTIONS,
    _build,
    _cands,
    _eval_paths,
    _extra,
    _fake_generate_chain,
    _HFStyleTokenizer,
    _paths,
    _qrow,
    _reworded,
    _round,
    _round_pool,
    _source_row,
    _Spend,
    _teacher_chain,
    _tiny_tokenizer,
    _train_row,
    _train_rows,
    _trow,
    _validate_tokens,
    _without_line,
    _write_test_set,
)

# ---- M1: formats and parsing ----
EOS = "<|end_of_text|>"
GOOD_CHAIN = (
    "Restate: A farmer has 17 sheep; all but 9 run away.\n"
    "Check: This is a wording trap.\n"
    "Solve: All but 9 means 9 remain.\n"
    "Verify: 17 - 8 = 9.\n"
    "Final answer: 9"
)


def _chain_without(label):
    return "\n".join(
        line for line in GOOD_CHAIN.split("\n") if not line.startswith(label)
    )


# ---- format_prompt ----
def test_format_prompt_exact_shape():
    expected = f"{config.INSTRUCTION_MARKER}What is 2+2?\n\n{config.RESPONSE_MARKER}"
    assert data.format_prompt("What is 2+2?") == expected


def test_format_prompt_ends_with_response_marker():
    assert data.format_prompt("Q?").endswith(config.RESPONSE_MARKER)


@pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
def test_format_prompt_rejects_empty_question(bad):
    with pytest.raises(ValueError):
        data.format_prompt(bad)


def test_format_prompt_strips_surrounding_whitespace():
    assert data.format_prompt("  hi?  \n") == data.format_prompt("hi?")


@pytest.mark.parametrize(
    "question",
    ["Line1\nLine2?", 'He said "hi" — didn’t he?', "Café naïve 日本語?"],
)
def test_format_prompt_preserves_inner_content(question):
    assert question in data.format_prompt(question)


@pytest.mark.parametrize("marker", [config.RESPONSE_MARKER, config.INSTRUCTION_MARKER])
def test_format_prompt_rejects_question_containing_marker(marker):
    with pytest.raises(ValueError):
        data.format_prompt(f"What is {marker} this?")


# ---- format_example ----
def test_format_example_is_prompt_plus_chain_plus_eos():
    expected = data.format_prompt("Q?") + GOOD_CHAIN + EOS
    assert data.format_example("Q?", GOOD_CHAIN, EOS) == expected


def test_format_example_eos_exactly_once():
    assert data.format_example("Q?", GOOD_CHAIN, EOS).count(EOS) == 1
    assert data.format_example("Q?", GOOD_CHAIN + EOS, EOS).count(EOS) == 1


def test_format_example_strips_trailing_whitespace():
    assert data.format_example("Q?", GOOD_CHAIN + "\n\n", EOS) == data.format_example(
        "Q?", GOOD_CHAIN, EOS
    )


def test_format_example_does_not_duplicate_final_answer():
    out = data.format_example("Q?", GOOD_CHAIN, EOS)
    assert out.count(config.FINAL_ANSWER_LABEL) == 1


@pytest.mark.parametrize("label", config.SECTION_LABELS)
def test_format_example_rejects_missing_section(label):
    with pytest.raises(ValueError):
        data.format_example("Q?", _chain_without(label), EOS)


def test_format_example_rejects_out_of_order_sections():
    lines = GOOD_CHAIN.split("\n")
    lines[1], lines[2] = lines[2], lines[1]
    with pytest.raises(ValueError):
        data.format_example("Q?", "\n".join(lines), EOS)


def test_format_example_rejects_duplicate_final_answer():
    with pytest.raises(ValueError):
        data.format_example("Q?", GOOD_CHAIN + "\nFinal answer: 10", EOS)


def test_format_example_rejects_empty_final_answer():
    chain = _chain_without("Final answer:") + "\nFinal answer:   "
    with pytest.raises(ValueError):
        data.format_example("Q?", chain, EOS)


def test_format_example_rejects_empty_chain():
    with pytest.raises(ValueError):
        data.format_example("Q?", "", EOS)


# ---- format_fewshot ----
EX1 = {"question": "How many months have 28 days?", "chain": GOOD_CHAIN}
EX2 = {
    "question": "What weighs more, a kilo of feathers or steel?",
    "chain": GOOD_CHAIN,
}


def test_format_fewshot_zero_examples_is_just_the_prompt():
    assert data.format_fewshot([], "Target?") == data.format_prompt("Target?")


def test_format_fewshot_exact_shape_and_order():
    expected = "\n\n".join(
        [
            data.format_prompt(EX1["question"]) + GOOD_CHAIN,
            data.format_prompt(EX2["question"]) + GOOD_CHAIN,
            data.format_prompt("Target?"),
        ]
    )
    assert data.format_fewshot([EX1, EX2], "Target?") == expected


def test_format_fewshot_is_stable():
    assert data.format_fewshot([EX1, EX2], "Target?") == data.format_fewshot(
        [EX1, EX2], "Target?"
    )


def test_format_fewshot_rejects_example_identical_to_target():
    with pytest.raises(ValueError):
        data.format_fewshot([EX1], "how many months have 28 DAYS")


def test_format_fewshot_length_limit_boundary():
    full = data.format_fewshot([EX1], "Target?")
    assert data.format_fewshot([EX1], "Target?", max_chars=len(full)) == full
    with pytest.raises(ValueError):
        data.format_fewshot([EX1], "Target?", max_chars=len(full) - 1)


# ---- extract_final_answer ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Restate: x\nFinal answer: 9", "9"),
        ("FINAL ANSWER: Nine", "Nine"),
        ("final answer: nine", "nine"),
        ("Final answer:    9   ", "9"),
        ("Final   answer : 9", "9"),
        ("  Final answer: 9", "9"),
        ("Final answer: 9\nFinal answer: 10", "9"),
        ("Restate: the Final answer: is tricky\nFinal answer: 9", "9"),
        ("Final answer: 9\n\nQuestion: What next?\nFinal answer: 12", "9"),
        ("Final answer: 9\n### Instruction:\nfoo", "9"),
        ("Final answer: It depends\non the day\n\nmore text", "It depends on the day"),
        ("Final answer: “Nobody”", "“Nobody”"),
    ],
)
def test_extract_final_answer_found(text, expected):
    assert data.extract_final_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Restate: nothing here",
        "Final answer is 9",
        "Restate: the Final answer: is tricky",
        "Final answer:",
        "Final answer:   \n",
    ],
)
def test_extract_final_answer_none(text):
    assert data.extract_final_answer(text) is None


# ---- normalize ----
@pytest.mark.parametrize(("raw", "expected"), [(None, ""), ("", ""), ("   ", "")])
def test_normalize_empty_inputs(raw, expected):
    assert data.normalize(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Hello   WORLD  ", "hello world"),
        ("a\tb\nc", "a b c"),
        ("Don’t stop!", "dont stop"),
        ("Don't stop!", "dont stop"),
        ("well-known", "well known"),
        ("“Nobody”", "nobody"),
        ("Nine.", "nine"),
        ("3.5", "3.5"),
        ("3.5.", "3.5"),
        ("ﬁsh", "fish"),
    ],
)
def test_normalize_cases(raw, expected):
    assert data.normalize(raw) == expected


def test_normalize_unicode_forms_agree():
    assert data.normalize("café") == data.normalize("café")


def test_normalize_decimal_not_collapsed():
    assert data.normalize("3.5") != data.normalize("35")


@pytest.mark.parametrize("raw", ["Don’t stop!", "  A  B ", "3.5.", None])
def test_normalize_is_idempotent(raw):
    once = data.normalize(raw)
    assert data.normalize(once) == once


# ---- is_correct ----
@pytest.mark.parametrize(
    ("predicted", "key", "alts"),
    [
        ("Nine.", "nine", []),
        ("9", "nine", ["9"]),
        ("  NINE ", "Nine", None),
    ],
)
def test_is_correct_true(predicted, key, alts):
    assert data.is_correct(predicted, key, alts or ())


@pytest.mark.parametrize(
    ("predicted", "key"),
    [
        ("not sure", "no"),
        ("nope", "no"),
        ("Yes, because it is", "yes"),
        ("A cat", "cat"),
        ("3.5", "35"),
        ("", "x"),
        (None, "x"),
    ],
)
def test_is_correct_false(predicted, key):
    assert not data.is_correct(predicted, key)


def test_is_correct_empty_alternative_does_not_match_empty_prediction():
    assert not data.is_correct("", "x", [""])


@pytest.mark.parametrize("key", ["", "   ", None])
def test_is_correct_rejects_empty_key(key):
    with pytest.raises(ValueError):
        data.is_correct("anything", key)


# ---- M1: extra edge cases (hostile / real-world variants) ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Restate: x\r\nFinal answer: 9\r\n", "9"),
        ("Final\tanswer:\t9", "9"),
        ("Final answer: 9", "9"),
        ("Final answer: 9\nRestate: next puzzle", "9"),
        ("Final answer: 9\nfinal answer: 10", "9"),
    ],
)
def test_extract_final_answer_extra_variants(text, expected):
    assert data.extract_final_answer(text) == expected


def test_format_example_accepts_indented_labels():
    chain = GOOD_CHAIN.replace("Final answer:", "  Final answer:")
    assert data.format_example("Q?", chain, EOS).endswith("9" + EOS)


def test_format_example_ignores_label_text_mid_line():
    chain = GOOD_CHAIN.replace("Check: This", "Check: Is this a Solve: trap? This")
    assert data.format_example("Q?", chain, EOS).count(EOS) == 1


def test_format_example_rejects_lowercase_labels():
    # extract_final_answer is case-insensitive; training chains must be exact
    chain = GOOD_CHAIN.replace("Final answer:", "final answer:")
    with pytest.raises(ValueError):
        data.format_example("Q?", chain, EOS)


def test_format_fewshot_rejects_malformed_example_chain():
    bad = {"question": "Other?", "chain": _chain_without("Verify:")}
    with pytest.raises(ValueError):
        data.format_fewshot([bad], "Target?")


def test_format_fewshot_rejects_marker_in_target():
    with pytest.raises(ValueError):
        data.format_fewshot([EX1], f"Target {config.RESPONSE_MARKER}")


def test_format_fewshot_keeps_input_order_not_sorted():
    assert data.format_fewshot([EX1, EX2], "T?") != data.format_fewshot(
        [EX2, EX1], "T?"
    )


def test_is_correct_ignores_punctuation_on_prediction():
    assert data.is_correct("Nine!", "nine")


def test_is_correct_unit_suffix_does_not_match_bare_number():
    assert not data.is_correct("5 km", "5")


def test_normalize_keeps_negative_sign():
    # '-5' and '5' are different answers; hyphen-as-minus must survive
    assert data.normalize("-5") != data.normalize("5")


def test_is_correct_negative_number_is_not_positive_number():
    assert not data.is_correct("-5", "5")


@pytest.mark.xfail(strict=True, reason="known gap: commas inside numbers")
def test_normalize_thousands_separator():
    assert data.normalize("1,000") == data.normalize("1000")


# ---- M1: marker injection class (question and chain share the template) ----
_BARE_MARKERS = [m.strip() for m in (config.INSTRUCTION_MARKER, config.RESPONSE_MARKER)]


@pytest.mark.parametrize("marker", _BARE_MARKERS)
@pytest.mark.parametrize("template", ["{m} tail", "head {m}", "head {m} tail", "{m}"])
def test_format_prompt_rejects_bare_marker_anywhere(marker, template):
    with pytest.raises(ValueError):
        data.format_prompt(template.format(m=marker))


@pytest.mark.parametrize("marker", _BARE_MARKERS)
def test_format_example_rejects_marker_in_chain(marker):
    chain = GOOD_CHAIN.replace("Solve: ", f"Solve: {marker} ")
    with pytest.raises(ValueError):
        data.format_example("Q?", chain, EOS)


def test_normalize_unicode_minus_equals_hyphen_minus():
    assert data.normalize("−5") == data.normalize("-5")


@pytest.mark.parametrize(
    ("raw", "expected"), [("2026-10-11", "2026 10 11"), ("-5", "-5")]
)
def test_normalize_hyphen_cases(raw, expected):
    assert data.normalize(raw) == expected


# ---- M2a: read_jsonl / write_jsonl ----
def test_read_jsonl_basic(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert data.read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_no_trailing_newline(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}', encoding="utf-8")
    assert data.read_jsonl(p) == [{"a": 1}]


def test_read_jsonl_skips_blank_lines(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('\n{"a": 1}\n\n   \n{"a": 2}\n\n', encoding="utf-8")
    assert data.read_jsonl(p) == [{"a": 1}, {"a": 2}]


def test_read_jsonl_handles_bom(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'\xef\xbb\xbf{"a": 1}\n')
    assert data.read_jsonl(p) == [{"a": 1}]


def test_read_jsonl_empty_file(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("", encoding="utf-8")
    assert data.read_jsonl(p) == []


def test_read_jsonl_malformed_line_names_physical_line_number(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text('{"a": 1}\n\n{oops}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        data.read_jsonl(p)


def test_read_jsonl_non_object_line_names_line_number(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        data.read_jsonl(p)


def test_read_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.read_jsonl(tmp_path / "nope.jsonl")


def test_write_jsonl_round_trip_keeps_unicode_and_newlines(tmp_path):
    rows = [{"q": "Café 日本語 “quotes”", "alts": ["a", "b"]}, {"q": "line\nbreak"}]
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, rows)
    assert data.read_jsonl(p) == rows


def test_write_jsonl_one_line_per_row_and_trailing_newline(tmp_path):
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, [{"q": "line\nbreak"}, {"q": "x"}])
    text = p.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(text.splitlines()) == 2


def test_write_jsonl_does_not_escape_unicode(tmp_path):
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, [{"q": "Café"}])
    assert "Café" in p.read_text(encoding="utf-8")


def test_write_jsonl_empty_rows_makes_empty_file(tmp_path):
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, [])
    assert p.read_text(encoding="utf-8") == ""
    assert data.read_jsonl(p) == []


def test_write_jsonl_creates_parent_directory(tmp_path):
    p = tmp_path / "sub" / "deeper" / "out.jsonl"
    data.write_jsonl(p, [{"a": 1}])
    assert data.read_jsonl(p) == [{"a": 1}]


def test_write_jsonl_overwrites_existing_file(tmp_path):
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, [{"a": 1}, {"a": 2}])
    data.write_jsonl(p, [{"a": 3}])
    assert data.read_jsonl(p) == [{"a": 3}]


def test_write_jsonl_failure_keeps_old_file_and_leaves_no_temp(tmp_path):
    p = tmp_path / "out.jsonl"
    data.write_jsonl(p, [{"a": 1}])
    old = p.read_text(encoding="utf-8")
    with pytest.raises(TypeError):
        data.write_jsonl(p, [{"a": 2}, {"a": object()}])
    assert p.read_text(encoding="utf-8") == old
    assert list(tmp_path.iterdir()) == [p]


# ---- M2a: check_schema ----
GOOD_ROW = {
    "id": "tqa-001",
    "question": "What happens if you eat watermelon seeds?",
    "answer": "Nothing happens",
    "accepted_alternatives": ["They pass through you"],
    "category": "misconception",
    "source": "truthfulqa",
}


def test_check_schema_accepts_good_row():
    assert data.check_schema(GOOD_ROW) is None


def test_check_schema_chain_is_optional_but_checked_when_present():
    assert data.check_schema({**GOOD_ROW, "chain": GOOD_CHAIN}) is None
    with pytest.raises(ValueError, match="chain"):
        data.check_schema({**GOOD_ROW, "chain": 5})


@pytest.mark.parametrize("key", list(GOOD_ROW))
def test_check_schema_rejects_missing_key(key):
    row = {k: v for k, v in GOOD_ROW.items() if k != key}
    with pytest.raises(ValueError, match=key):
        data.check_schema(row)


@pytest.mark.parametrize("key", ["id", "question", "answer", "source"])
@pytest.mark.parametrize("bad", ["", "   ", None, 5])
def test_check_schema_rejects_empty_or_non_string_text(key, bad):
    with pytest.raises(ValueError, match=key):
        data.check_schema({**GOOD_ROW, key: bad})


def test_check_schema_rejects_unknown_category():
    with pytest.raises(ValueError, match="category"):
        data.check_schema({**GOOD_ROW, "category": "character_counting"})


@pytest.mark.parametrize("bad", ["nine", ("nine",), None, {"a": 1}])
def test_check_schema_alternatives_must_be_a_list(bad):
    with pytest.raises(ValueError, match="accepted_alternatives"):
        data.check_schema({**GOOD_ROW, "accepted_alternatives": bad})


@pytest.mark.parametrize("bad", [[""], ["  "], [None], [5], ["ok", ""]])
def test_check_schema_alternatives_items_must_be_non_empty_strings(bad):
    with pytest.raises(ValueError, match="accepted_alternatives"):
        data.check_schema({**GOOD_ROW, "accepted_alternatives": bad})


# ---- M2a: stratified_sample (n is PER CATEGORY) ----
def _rows(counts):
    return [
        {"id": f"{cat}-{i}", "category": cat}
        for cat, size in counts.items()
        for i in range(size)
    ]


def _count(rows, cat):
    return sum(r["category"] == cat for r in rows)


def test_stratified_sample_takes_n_per_category():
    out = data.stratified_sample(_rows({"a": 10, "b": 10}), 3, seed=0)
    assert _count(out, "a") == 3
    assert _count(out, "b") == 3


def test_stratified_sample_small_category_returns_all_without_error():
    out = data.stratified_sample(_rows({"a": 10, "b": 2}), 5, seed=0)
    assert _count(out, "a") == 5
    assert sorted(r["id"] for r in out if r["category"] == "b") == ["b-0", "b-1"]


def test_stratified_sample_single_row_category_is_kept():
    out = data.stratified_sample(_rows({"a": 10, "solo": 1}), 3, seed=0)
    assert _count(out, "solo") == 1


def test_stratified_sample_n_larger_than_everything_returns_all():
    rows = _rows({"a": 3, "b": 2})
    out = data.stratified_sample(rows, 100, seed=0)
    assert sorted(r["id"] for r in out) == sorted(r["id"] for r in rows)


def test_stratified_sample_has_no_duplicates():
    out = data.stratified_sample(_rows({"a": 10, "b": 10}), 7, seed=3)
    ids = [r["id"] for r in out]
    assert len(ids) == len(set(ids))


def test_stratified_sample_empty_input():
    assert data.stratified_sample([], 3, seed=0) == []


def test_stratified_sample_n_zero_is_empty():
    assert data.stratified_sample(_rows({"a": 5}), 0, seed=0) == []


def test_stratified_sample_negative_n_raises():
    with pytest.raises(ValueError):
        data.stratified_sample(_rows({"a": 5}), -1, seed=0)


def test_stratified_sample_same_seed_same_result():
    rows = _rows({"a": 20, "b": 20})
    assert data.stratified_sample(rows, 4, seed=7) == data.stratified_sample(
        rows, 4, seed=7
    )


def test_stratified_sample_different_seed_changes_result():
    rows = _rows({"a": 20, "b": 20})
    assert data.stratified_sample(rows, 4, seed=1) != data.stratified_sample(
        rows, 4, seed=2
    )


def test_stratified_sample_does_not_mutate_input():
    rows = _rows({"a": 6, "b": 6})
    before = [dict(r) for r in rows]
    data.stratified_sample(rows, 2, seed=0)
    assert rows == before


def test_stratified_sample_row_without_category_raises():
    with pytest.raises(ValueError, match="category"):
        data.stratified_sample([{"id": "x"}], 1, seed=0)


def test_written_rows_keep_every_schema_field_and_a_chain(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    for row in data.read_jsonl(out):
        data.check_schema(row)
        assert row["chain"].startswith("CHAIN-")


# ---- drop_excluded ----
def test_drop_excluded_removes_by_id_even_when_the_text_differs():
    rows = [_trow(1), _trow(2, id="t-sphinx", question="Reworded sphinx riddle?")]
    assert [r["id"] for r in data.drop_excluded(rows, {"t-sphinx"})] == ["t-1"]


def test_drop_excluded_keeps_a_similar_looking_row_with_another_id():
    rows = [_trow(1, question="What walks on four legs, the sphinx riddle?")]
    assert data.drop_excluded(rows, {"t-sphinx"}) == rows


def test_drop_excluded_with_nothing_excluded_returns_all_rows_as_a_new_list():
    rows = [_trow(1), _trow(2)]
    kept = data.drop_excluded(rows, set())
    assert kept == rows
    assert kept is not rows


# ---- check_test_set ----
def test_check_test_set_accepts_good_rows():
    assert data.check_test_set([_trow(1), _trow(2)]) is None


def test_check_test_set_rejects_no_rows():
    with pytest.raises(ValueError, match="no rows"):
        data.check_test_set([])


def test_check_test_set_schema_failure_names_the_row():
    with pytest.raises(ValueError, match="t-1"):
        data.check_test_set([_trow(1, category="not_a_category")])


def test_check_test_set_requires_a_chain():
    row = {k: v for k, v in _trow(1).items() if k != "chain"}
    with pytest.raises(ValueError, match="chain"):
        data.check_test_set([row])


def test_check_test_set_rejects_a_chain_missing_a_section():
    chain = _without_line(_teacher_chain("9"), "Verify:")
    with pytest.raises(ValueError, match="t-1"):
        data.check_test_set([_trow(1, chain=chain)])


def test_check_test_set_final_answer_must_equal_the_key():
    with pytest.raises(ValueError, match="final answer"):
        data.check_test_set([_trow(1, chain=_teacher_chain("12"))])


def test_check_test_set_final_answer_may_equal_an_alternative():
    assert data.check_test_set([_trow(1, chain=_teacher_chain("nine"))]) is None


def test_check_test_set_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        data.check_test_set([_trow(1), _trow(2, id="t-1")])


def test_check_test_set_time_sensitive_needs_the_reference_year():
    row = _trow(1, category="time_sensitive")
    with pytest.raises(ValueError, match="reference_year"):
        data.check_test_set([row])
    with pytest.raises(ValueError, match="reference_year"):
        data.check_test_set([{**row, "reference_year": config.REFERENCE_YEAR - 1}])
    assert (
        data.check_test_set([{**row, "reference_year": config.REFERENCE_YEAR}]) is None
    )


def test_check_test_set_reference_year_is_read_at_call_time(monkeypatch):
    row = _trow(1, category="time_sensitive", reference_year=2026)
    monkeypatch.setattr(config, "REFERENCE_YEAR", 2027)
    with pytest.raises(ValueError, match="reference_year"):
        data.check_test_set([row])


def test_check_test_set_other_categories_need_no_reference_year():
    assert data.check_test_set([_trow(1, category="false_premise")]) is None


def test_load_test_set_defaults_the_source_and_keeps_a_given_one(tmp_path):
    no_source = {k: v for k, v in _trow(1).items() if k != "source"}
    path = _write_test_set(tmp_path, [no_source, _trow(2, source="class")])
    rows = data.load_test_set(path, excluded=set())
    assert [r["source"] for r in rows] == ["hand_written", "class"]


def test_load_test_set_drops_excluded_ids_then_validates(tmp_path):
    bad_but_excluded = _trow(9, id="t-sphinx", category="not_a_category")
    path = _write_test_set(tmp_path, [_trow(1), bad_but_excluded])
    rows = data.load_test_set(path, excluded={"t-sphinx"})
    assert [r["id"] for r in rows] == ["t-1"]


def test_load_test_set_validation_failure_stops(tmp_path):
    path = _write_test_set(tmp_path, [_trow(1), _trow(2, chain=_teacher_chain("12"))])
    with pytest.raises(ValueError, match="t-2"):
        data.load_test_set(path, excluded=set())


def test_load_test_set_everything_excluded_raises(tmp_path):
    path = _write_test_set(tmp_path, [_trow(1)])
    with pytest.raises(ValueError, match="no rows"):
        data.load_test_set(path, excluded={"t-1"})


def test_load_test_set_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_test_set(tmp_path / "nope.jsonl", excluded=set())


def test_load_test_set_reads_config_defaults_at_call_time(monkeypatch, tmp_path):
    path = _write_test_set(tmp_path, [_trow(1), _trow(2, id="t-sphinx")])
    monkeypatch.setattr(config, "TEST_SET_PATH", path)
    monkeypatch.setattr(config, "EXCLUDED_TEST_IDS", frozenset({"t-sphinx"}))
    assert [r["id"] for r in data.load_test_set()] == ["t-1"]


# ---- check_leakage ----
def test_check_leakage_clean_returns_no_warnings(recwarn):
    assert data.check_leakage([_trow(1)], [_train_row(1), _train_row(2)]) == []
    assert len(recwarn) == 0


def test_check_leakage_empty_train_raises():
    with pytest.raises(ValueError, match="empty"):
        data.check_leakage([_trow(1)], [])


def test_check_leakage_duplicate_test_ids_raise():
    with pytest.raises(ValueError, match="duplicate"):
        data.check_leakage([_trow(1), _trow(2, id="t-1")], [_train_row(1)])


def test_check_leakage_same_question_in_train_raises_naming_both_ids():
    train = [_train_row(1, question="Hand written question number 1?")]
    with pytest.raises(ValueError, match="t-1.*tr-1"):
        data.check_leakage([_trow(1)], train)


def test_check_leakage_same_question_differing_in_case_and_punctuation_raises():
    train = [_train_row(1, question="HAND written question number 1")]
    with pytest.raises(ValueError, match="t-1"):
        data.check_leakage([_trow(1)], train)


def test_check_leakage_identical_chain_raises_even_with_whitespace_differences():
    train = [_train_row(1, chain="\n" + _trow(1)["chain"] + "\n\n")]
    with pytest.raises(ValueError, match="chain"):
        data.check_leakage([_trow(1)], train)


def test_check_leakage_eval_variant_in_train_raises():
    eval_rows = [{**_trow(1), "id": "e-1", "question": "A reworded variant of one?"}]
    train = [_train_row(1, question="a reworded VARIANT of one")]
    with pytest.raises(ValueError, match="eval.*e-1"):
        data.check_leakage([_trow(1)], train, eval_rows)


def test_check_leakage_eval_variants_not_in_train_are_fine():
    eval_rows = [{**_trow(1), "id": "e-1", "question": "A reworded variant of one?"}]
    assert data.check_leakage([_trow(1)], [_train_row(1)], eval_rows) == []


def test_check_leakage_near_duplicate_warns_and_is_returned():
    test = [
        _trow(
            1, question="What is the capital of France and why is it famous worldwide?"
        )
    ]
    train = [
        _train_row(
            1, question="What is the capital of France and why is it famous world wide?"
        )
    ]
    with pytest.warns(UserWarning, match="t-1"):
        found = data.check_leakage(test, train)
    assert len(found) == 1
    assert (found[0]["test_id"], found[0]["train_id"]) == ("t-1", "tr-1")
    assert found[0]["ratio"] >= config.NEAR_DUPLICATE_RATIO


def test_check_leakage_clearly_different_questions_do_not_warn(recwarn):
    test = [_trow(1, question="How many sheep are left after the farmer sells nine?")]
    train = [_train_row(1, question="What is the boiling point of water at sea level?")]
    assert data.check_leakage(test, train) == []
    assert len(recwarn) == 0


# ---- test-set hash ----
def test_file_sha256_matches_a_known_value(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    assert data.file_sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_record_test_set_hash_stores_sha_and_row_count(tmp_path):
    test_path = _write_test_set(tmp_path, [_trow(1), _trow(2)])
    hash_path = tmp_path / "hash.jsonl"
    data.record_test_set_hash(test_path, hash_path)
    (record,) = data.read_jsonl(hash_path)
    assert record == {"sha256": data.file_sha256(test_path), "rows": 2}


def test_check_test_set_hash_ok_changed_and_missing(tmp_path, recwarn):
    test_path = _write_test_set(tmp_path, [_trow(1)])
    hash_path = tmp_path / "hash.jsonl"
    assert data.check_test_set_hash(test_path, hash_path) == "missing"
    assert len(recwarn) == 0
    data.record_test_set_hash(test_path, hash_path)
    assert data.check_test_set_hash(test_path, hash_path) == "ok"
    assert len(recwarn) == 0
    data.write_jsonl(test_path, [_trow(1), _trow(2)])
    with pytest.warns(UserWarning, match="changed"):
        assert data.check_test_set_hash(test_path, hash_path) == "changed"


def test_record_test_set_hash_overwrites_the_old_record(tmp_path):
    test_path = _write_test_set(tmp_path, [_trow(1)])
    hash_path = tmp_path / "hash.jsonl"
    data.record_test_set_hash(test_path, hash_path)
    data.write_jsonl(test_path, [_trow(1), _trow(2)])
    data.record_test_set_hash(test_path, hash_path)
    assert data.check_test_set_hash(test_path, hash_path) == "ok"


def test_check_test_set_hash_missing_test_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.check_test_set_hash(tmp_path / "nope.jsonl", tmp_path / "hash.jsonl")


# ---- M4: drop_leaked (training rows that duplicate a test question) ----
def test_drop_leaked_removes_an_exact_question_ignoring_case_and_punctuation():
    train = [
        _train_row(1, question="What happens if you crack your knuckles a lot?"),
        _train_row(2),
    ]
    test = [_trow(1, question="what happens if you CRACK your knuckles a lot")]
    with pytest.warns(UserWarning, match="tr-1"):
        kept, dropped = data.drop_leaked(train, test)
    assert [r["id"] for r in kept] == ["tr-2"]
    assert dropped == ["tr-1"]


def test_drop_leaked_removes_a_near_duplicate():
    train = [
        _train_row(
            1, question="What is the capital of France and why is it famous world wide?"
        )
    ]
    test = [
        _trow(
            1, question="What is the capital of France and why is it famous worldwide?"
        )
    ]
    with pytest.warns(UserWarning):
        kept, dropped = data.drop_leaked(train, test)
    assert kept == []
    assert dropped == ["tr-1"]


def test_drop_leaked_keeps_everything_when_nothing_leaks(recwarn):
    train = [_train_row(1), _train_row(2)]
    kept, dropped = data.drop_leaked(train, [_trow(1)])
    assert kept == train
    assert dropped == []
    assert len(recwarn) == 0


def test_drop_leaked_with_no_test_rows_keeps_everything():
    train = [_train_row(1)]
    assert data.drop_leaked(train, []) == (train, [])


def test_drop_leaked_does_not_mutate_the_input():
    train = [_train_row(1, question="Hand written question number 1?"), _train_row(2)]
    before = list(train)
    with pytest.warns(UserWarning):
        data.drop_leaked(train, [_trow(1)])
    assert train == before


def test_variants_carry_source_id_category_answer_and_reference_chain():
    row = _source_row(category="false_premise")
    out = data.make_variants(row, [_reworded("perform")], cap=3)
    assert len(out) == 1
    variant = out[0]
    assert variant["id"] == "t-1-v0"
    assert variant["source_id"] == "t-1"
    assert variant["category"] == "false_premise"
    assert variant["answer"] == row["answer"]
    assert variant["accepted_alternatives"] == row["accepted_alternatives"]
    assert variant["chain"] == row["chain"]
    assert variant["question"] == _reworded("perform")
    assert variant["source"] == "paraphrase"
    data.check_schema(variant)


def test_variants_drop_unchanged_and_whitespace_case_or_punctuation_only_changes():
    good = _reworded("perform")
    candidates = [
        LONG_Q,
        LONG_Q + "  ",
        LONG_Q.upper(),
        LONG_Q.replace(",", " ,"),
        good,
    ]
    out = data.make_variants(_source_row(), candidates, cap=3)
    assert [v["question"] for v in out] == [good]


def test_variants_drop_duplicates_keeping_the_first():
    a = _reworded("perform")
    b = LONG_Q.replace("opera", "operetta")
    out = data.make_variants(_source_row(), [a, b, a, b.upper()], cap=3)
    assert [v["question"] for v in out] == [a, b]


def test_variants_enforce_the_cap_and_keep_the_order():
    candidates = [_reworded(w) for w in ("perform", "act", "stage", "sing", "do")]
    out = data.make_variants(_source_row(), candidates, cap=2)
    assert [v["question"] for v in out] == candidates[:2]


@pytest.mark.parametrize(
    "bad",
    [
        LONG_Q.replace("600", "six hundred"),
        LONG_Q.replace("600", "60"),
        LONG_Q.replace("1 hour", "1 minute"),
        LONG_Q.replace("1 hour", "i hour"),
    ],
    ids=["digit-to-word", "digit-changed", "unit-changed", "number-as-letter"],
)
def test_variants_drop_changed_numbers_and_units(bad):
    assert data.make_variants(_source_row(), [bad], cap=3) == []


def test_variants_allow_a_plural_unit_with_the_same_number():
    plural = LONG_Q.replace("1 hour for", "1 hours for")
    out = data.make_variants(_source_row(), [plural], cap=3)
    assert [v["question"] for v in out] == [plural]


@pytest.mark.parametrize(
    ("question", "bad"),
    [
        (
            "Is it safe to go upstairs if there is not a shark in the pool?",
            "Is it safe to go upstairs if there is a shark in the pool?",
        ),
        (
            "Is it safe to go upstairs if there is a shark in the pool?",
            "Is it safe to go upstairs if there is not a shark in the pool?",
        ),
        (
            "Why can't a person with no shoes walk on hot sand today?",
            "Why can a person with no shoes walk on hot sand today?",
        ),
        (
            "Why does nobody never answer the door when I knock loudly?",
            "Why does nobody ever answer the door when I knock loudly?",
        ),
    ],
)
def test_variants_drop_dropped_or_added_negations(question, bad):
    row = _trow(1, question=question)
    assert data.make_variants(row, [bad], cap=3) == []


def test_variants_short_questions_yield_none():
    row = _trow(1, question="Was Talos alive?")
    assert data.make_variants(row, ["Was Talos living?"], cap=3) == []


def test_variants_skip_blank_and_non_string_candidates():
    good = _reworded("perform")
    out = data.make_variants(_source_row(), ["", "   ", None, 5, good], cap=3)
    assert [v["question"] for v in out] == [good]


def test_variants_skip_candidates_containing_a_template_marker():
    bad = _reworded("perform") + " ### Response:"
    good = _reworded("act")
    out = data.make_variants(_source_row(), [bad, good], cap=3)
    assert [v["question"] for v in out] == [good]


def test_variants_with_no_candidates_is_an_empty_list():
    assert data.make_variants(_source_row(), [], cap=3) == []


def test_variant_ids_count_the_kept_variants_of_each_question():
    a = _reworded("perform")
    b = LONG_Q.replace("opera", "operetta")
    out = data.make_variants(_source_row(), [LONG_Q, a, b], cap=3)
    assert [v["id"] for v in out] == ["t-1-v0", "t-1-v1"]


def test_a_higher_target_later_judges_cached_candidates_before_asking_for_more(
    tmp_path, monkeypatch
):
    _build(tmp_path, monkeypatch, [_qrow(1)], per_question=1)
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=2)
    assert spend.asked == []
    assert len(spend.judged) == 1
    out, _, _ = _eval_paths(tmp_path)
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v0", "t-1-v1"]
    assert summary["kept"] == 2


def test_a_lower_target_later_uses_the_cache_and_spends_nothing(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, [_qrow(1)], per_question=3)
    _, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=1)
    out, _, _ = _eval_paths(tmp_path)
    assert spend.asked == [] and spend.judged == []
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v0"]


def test_review_file_records_the_reason(tmp_path, monkeypatch):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    spend = _Spend({pool[0]: {"keep": False, "reason": "answer_changed"}})
    summary, _ = _build(tmp_path, monkeypatch, [_qrow(1)], spend)
    _, review, _ = _eval_paths(tmp_path)
    (rejected,) = data.read_jsonl(review)
    assert rejected["verdict"] == {"keep": False, "reason": "answer_changed"}
    assert summary["rejected_by_reason"] == {"answer_changed": 1}


def test_a_failed_judge_call_is_not_cached_and_is_retried_next_run(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    summary, _ = _build(tmp_path, monkeypatch, [_qrow(1)], _Spend(errors={pool[0]}))
    out, review, _ = _eval_paths(tmp_path)
    assert summary["judge_errors"] == 1
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v1", "t-1-v2"]
    assert data.read_jsonl(review) == []
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], _Spend())
    assert spend.judged == [pool[0]]
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v0", "t-1-v1"]


def test_the_eval_files_are_rewritten_freely_because_it_costs_nothing(
    tmp_path, monkeypatch
):
    out, _, _ = _eval_paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    _build(tmp_path, monkeypatch, [_qrow(1)])
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v0", "t-1-v1"]


# ---- M5: pick_random (the printout of random variants) ----
def test_pick_random_returns_n_rows_in_original_order_and_is_seeded():
    rows = [{"id": i} for i in range(30)]
    first = data.pick_random(rows, 20, seed=1)
    assert len(first) == 20
    assert first == data.pick_random(rows, 20, seed=1)
    assert first != data.pick_random(rows, 20, seed=2)
    assert [r["id"] for r in first] == sorted(r["id"] for r in first)


def test_pick_random_with_fewer_rows_than_requested_returns_all():
    rows = [{"id": i} for i in range(3)]
    assert data.pick_random(rows, 20, seed=0) == rows
    assert data.pick_random([], 20, seed=0) == []


# ---- stratified_sample is prefix-stable: a bigger n contains the smaller sample ----
@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("small", [1, 3, 10])
def test_stratified_sample_larger_n_contains_the_smaller_sample(seed, small):
    rows = _rows({"a": 40, "b": 25, "c": 6})
    few = {r["id"] for r in data.stratified_sample(rows, small, seed=seed)}
    many = {r["id"] for r in data.stratified_sample(rows, small + 12, seed=seed)}
    assert few <= many


def test_stratified_sample_stays_stable_when_other_categories_change_size():
    base = _rows({"a": 40, "b": 25})
    more = _rows({"a": 40, "b": 60})
    pick_a = lambda rows: {
        r["id"] for r in data.stratified_sample(rows, 5, seed=3) if r["category"] == "a"
    }
    assert pick_a(base) == pick_a(more)


def _tok_len(tok, text, special=False):
    return len(tok.encode(text, add_special_tokens=special).ids)


def test_load_tokenizer_from_a_local_file(tmp_path):
    path = tmp_path / "tok.json"
    _tiny_tokenizer().save(str(path))
    loaded = data.load_tokenizer(str(path))
    assert _tok_len(loaded, "a b c") == 3


def test_token_lengths_prompt_matches_a_hand_count_including_the_start_token():
    # "### Instruction : a b ### Response :" is 8 words, plus <bos> = 9
    row = _trow(1, question="a b", chain=_teacher_chain("9"))
    (length,) = data.token_lengths([row], _tiny_tokenizer())
    assert length["prompt"] == 9


def test_token_lengths_start_token_is_counted_only_when_the_tokenizer_adds_it():
    row = _trow(1, question="a b")
    with_bos = data.token_lengths([row], _tiny_tokenizer(with_bos=True))[0]
    without = data.token_lengths([row], _tiny_tokenizer(with_bos=False))[0]
    assert with_bos["prompt"] == without["prompt"] + 1
    assert with_bos["total"] == without["total"] + 1


def test_token_lengths_response_counts_the_chain_and_one_end_token():
    tok = _tiny_tokenizer()
    chain = _teacher_chain("9")
    row = _trow(1, question="a b", chain=chain)
    (length,) = data.token_lengths([row], tok)
    assert length["response"] == _tok_len(tok, chain) + 1
    assert length["total"] == length["prompt"] + length["response"]


def test_token_lengths_report_prompt_and_response_separately_per_row():
    tok = _tiny_tokenizer()
    short = _trow(1, question="a b", chain=_teacher_chain("9"))
    long = _trow(
        2, question="a b c d e f g h", chain=_teacher_chain("9", "extra words " * 8)
    )
    first, second = data.token_lengths([short, long], tok)
    assert second["prompt"] == first["prompt"] + 6
    assert second["response"] > first["response"]
    assert [first["id"], second["id"]] == ["t-1", "t-2"]


def test_token_lengths_row_without_a_chain_names_the_row():
    row = {k: v for k, v in _trow(1).items() if k != "chain"}
    with pytest.raises(ValueError, match="t-1"):
        data.token_lengths([row], _tiny_tokenizer())


def test_token_lengths_of_no_rows_is_empty():
    assert data.token_lengths([], _tiny_tokenizer()) == []


def test_token_lengths_uses_the_configured_end_token_by_default(monkeypatch):
    tok = _tiny_tokenizer()
    tok.add_special_tokens(["<other_eos>"])
    row = _trow(1, question="a b")
    default = data.token_lengths([row], tok)[0]
    explicit = data.token_lengths([row], tok, eos_token="<other_eos>")[0]
    assert default["total"] == explicit["total"]


def _lengths(*totals):
    return [
        {"id": f"x-{i}", "prompt": t // 4, "response": t - t // 4, "total": t}
        for i, t in enumerate(totals)
    ]


def test_length_summary_reports_prompt_response_and_total_separately():
    summary = data.length_summary(_lengths(10, 20, 30, 40, 100))
    assert summary["n"] == 5
    assert summary["max_total"] == 100
    assert summary["total"]["p50"] == 30
    assert summary["total"]["p90"] == pytest.approx(76.0)
    assert summary["total"]["max"] == 100
    assert summary["prompt"]["max"] == 25
    assert summary["response"]["max"] == 75


def test_length_summary_with_one_example_has_equal_percentiles():
    summary = data.length_summary(_lengths(42))
    assert summary["total"]["p50"] == summary["total"]["p90"] == 42
    assert summary["total"]["p99"] == summary["total"]["max"] == 42


def test_length_summary_lists_the_longest_examples_first():
    summary = data.length_summary(_lengths(10, 50, 30, 40))
    assert summary["longest"][:3] == [
        {"id": "x-1", "total": 50},
        {"id": "x-3", "total": 40},
        {"id": "x-2", "total": 30},
    ]


def test_length_summary_of_nothing_raises():
    with pytest.raises(ValueError, match="no examples"):
        data.length_summary([])


def test_check_token_limit_accepts_an_example_exactly_at_the_limit():
    assert data.check_token_limit(_lengths(10, 64), 64) is None


def test_check_token_limit_rejects_an_example_over_the_limit_naming_it():
    with pytest.raises(ValueError, match="x-1") as caught:
        data.check_token_limit(_lengths(10, 65, 20), 64)
    assert "65" in str(caught.value)
    assert "1 example" in str(caught.value)


def test_check_token_limit_counts_every_example_over_the_limit():
    with pytest.raises(ValueError, match="3 examples"):
        data.check_token_limit(_lengths(70, 80, 90, 10), 64)


def test_check_token_limit_of_nothing_raises():
    with pytest.raises(ValueError, match="no examples"):
        data.check_token_limit([], 64)


@pytest.mark.parametrize(
    ("longest", "expected"),
    [
        (1, 64),
        (64, 64),
        (65, 128),
        (227, 256),
        (1000, 1024),
        (1024, 1024),
        (1025, 1088),
    ],
)
def test_recommended_max_seq_length_is_the_next_multiple_of_64(longest, expected):
    assert data.recommended_max_seq_length(longest) == expected


@pytest.mark.parametrize("bad", [0, -5])
def test_recommended_max_seq_length_rejects_non_positive_lengths(bad):
    with pytest.raises(ValueError):
        data.recommended_max_seq_length(bad)


@pytest.mark.slow
def test_real_llama_tokenizer_counts_start_and_end_as_single_tokens():
    tok = data.load_tokenizer(config.MODEL_NAME)
    with_special = tok.encode("hi" + config.EOS_TOKEN, add_special_tokens=True)
    assert with_special.ids[0] == tok.token_to_id("<|begin_of_text|>")
    assert with_special.ids[-1] == tok.token_to_id(config.EOS_TOKEN)
    assert with_special.ids.count(tok.token_to_id(config.EOS_TOKEN)) == 1


@pytest.mark.slow
def test_real_llama_tokenizer_keeps_the_prompt_as_an_exact_prefix_of_the_example():
    tok = data.load_tokenizer(config.MODEL_NAME)
    chain = _teacher_chain("9")
    prompt = data.format_prompt(LONG_Q)
    full = data.format_example(LONG_Q, chain, config.EOS_TOKEN)
    prompt_ids = tok.encode(prompt, add_special_tokens=True).ids
    full_ids = tok.encode(full, add_special_tokens=True).ids
    assert full_ids[: len(prompt_ids)] == prompt_ids


def test_pipeline_validate_tokens_passes_and_writes_the_audit(tmp_path):
    result = _validate_tokens(tmp_path, "--max-seq-length", "1000")
    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["train"]["n"] == 2
    assert audit["max_seq_length"] == 1000
    assert audit["recommended_max_seq_length"] % 64 == 0
    assert "max" in result.stdout


def test_pipeline_validate_tokens_fails_when_the_limit_is_below_the_longest(tmp_path):
    result = _validate_tokens(tmp_path, "--max-seq-length", "5")
    assert result.returncode != 0
    assert "tr-" in result.stderr
    assert "recommended" in result.stderr


def test_pipeline_validate_tokens_reports_eval_and_test_sets_separately(tmp_path):
    eval_rows = [
        {**_trow(1), "id": "e-1", "question": "A reworded variant of question one?"}
    ]
    result = _validate_tokens(tmp_path, "--max-seq-length", "1000", eval_rows=eval_rows)
    assert result.returncode == 0, result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["eval"]["n"] == 1
    assert audit["test"]["n"] == 1


def test_pipeline_validate_tokens_prints_the_memory_estimate_and_cheapest_gpu(tmp_path):
    result = _validate_tokens(tmp_path, "--max-seq-length", "1000")
    assert "GiB" in result.stdout
    assert "T4" in result.stdout


def test_pipeline_validate_tokens_fails_when_the_estimate_does_not_fit_the_gpu(
    tmp_path,
):
    result = _validate_tokens(
        tmp_path, "--max-seq-length", "1000", "--gpu", "T4", "--batch-size", "512"
    )
    assert result.returncode != 0
    assert "does not fit" in result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["fits_chosen_gpu"] is False


def test_pipeline_validate_tokens_names_a_bigger_gpu_when_the_chosen_one_is_too_small(
    tmp_path,
):
    result = _validate_tokens(
        tmp_path, "--max-seq-length", "1000", "--gpu", "T4", "--batch-size", "128"
    )
    assert result.returncode != 0
    assert "does not fit" in result.stderr
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["cheapest_gpu"] not in (None, "T4")


def test_pipeline_validate_tokens_when_no_gpu_fits_says_so_and_still_writes_the_audit(
    tmp_path,
):
    result = _validate_tokens(
        tmp_path, "--max-seq-length", "1000", "--batch-size", "100000"
    )
    assert result.returncode != 0
    assert "Traceback" not in result.stderr
    assert "no GPU" in result.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["cheapest_gpu"] is None


def test_encode_ids_gives_the_same_list_for_both_tokenizer_styles():
    inner = _tiny_tokenizer()
    assert data.encode_ids(inner, "a b c") == data.encode_ids(
        _HFStyleTokenizer(inner), "a b c"
    )
    assert isinstance(data.encode_ids(_HFStyleTokenizer(inner), "a b c"), list)


def test_token_lengths_work_with_a_transformers_style_tokenizer():
    inner = _tiny_tokenizer()
    rows = [_trow(1, question="a b"), _trow(2, question="c d e f")]
    assert data.token_lengths(rows, _HFStyleTokenizer(inner)) == data.token_lengths(
        rows, inner
    )


def _ids_of(result):
    return [r["id"] for r in result["rows"]]


def test_round_with_extras_mixes_the_new_rows_with_a_replay_of_other_categories():
    result = _round()
    ids = _ids_of(result)
    assert sum(i.startswith("extra-") for i in ids) == 4
    assert result["new"] == 4 and result["replay"] == 4
    assert len(ids) == 8


def test_replay_rows_never_come_from_a_weak_category_and_are_not_repeated():
    result = _round(replay_fraction=2.0)
    replay = [r for r in result["rows"] if not r["id"].startswith("extra-")]
    assert all(r["category"] != "pun_ambiguity" for r in replay)
    assert len({r["id"] for r in replay}) == len(replay) == 8


def test_replay_fraction_zero_means_only_the_new_rows():
    result = _round(replay_fraction=0)
    assert result["replay"] == 0
    assert sorted(_ids_of(result)) == ["extra-1", "extra-2", "extra-3", "extra-4"]


def test_replay_fraction_above_one_replays_more_than_the_new_rows():
    result = _round(replay_fraction=2.5)
    assert result["new"] == 4
    assert result["replay"] == 10


def test_a_replay_larger_than_the_available_data_uses_everything_once():
    result = _round(replay_fraction=100)
    others = [r for r in _round_pool() if r["category"] != "pun_ambiguity"]
    assert result["replay"] == len(others)
    assert len(result["rows"]) == len(others) + 4


def test_a_negative_replay_fraction_is_rejected():
    with pytest.raises(ValueError, match="replay_fraction"):
        _round(replay_fraction=-0.5)


def test_without_extras_the_existing_weak_rows_are_repeated_up_to_the_cap():
    for none_like in (None, []):
        result = _round(extra_rows=none_like, repeat_cap=2, replay_fraction=0)
        counts = {}
        for row in result["rows"]:
            counts[row["id"]] = counts.get(row["id"], 0) + 1
        assert set(counts.values()) == {2}
        assert result["new"] == 5 * 2
        assert all(row["category"] == "pun_ambiguity" for row in result["rows"])


def test_the_repeat_cap_is_never_exceeded():
    result = _round(extra_rows=None, repeat_cap=1, replay_fraction=0)
    ids = _ids_of(result)
    assert len(ids) == len(set(ids)) == 5


@pytest.mark.parametrize("cap", [0, -1])
def test_a_repeat_cap_below_one_is_rejected(cap):
    with pytest.raises(ValueError, match="repeat_cap"):
        _round(extra_rows=None, repeat_cap=cap)


def test_a_weak_category_with_no_training_rows_and_no_extras_is_rejected():
    pool = [r for r in _round_pool() if r["category"] != "pun_ambiguity"]
    with pytest.raises(ValueError, match="pun_ambiguity"):
        _round(train_rows=pool, extra_rows=None)


def test_a_weak_category_missing_from_training_is_fine_when_extras_are_given():
    pool = [r for r in _round_pool() if r["category"] != "pun_ambiguity"]
    assert _round(train_rows=pool)["new"] == 4


def test_an_unknown_weak_category_is_rejected():
    with pytest.raises(ValueError, match="character_counting"):
        _round(weak_categories=["character_counting"])


def test_an_empty_weak_category_list_is_rejected():
    with pytest.raises(ValueError, match="no weak categories"):
        _round(weak_categories=[])


def test_extras_with_an_unknown_category_are_rejected_naming_the_row():
    with pytest.raises(ValueError, match="extra-2"):
        _round(extra_rows=[_extra(1), _extra(2, category="character_counting")])


def test_extras_without_a_chain_or_with_a_broken_chain_are_rejected_naming_the_row():
    no_chain = {k: v for k, v in _extra(2).items() if k != "chain"}
    with pytest.raises(ValueError, match="extra-2"):
        _round(extra_rows=[_extra(1), no_chain])
    with pytest.raises(ValueError, match="extra-3"):
        _round(
            extra_rows=[
                _extra(1),
                _extra(3, chain=_without_line(_teacher_chain("a"), "Verify:")),
            ]
        )


def test_an_extra_that_duplicates_an_existing_training_question_is_dropped_with_a_warning():
    pool = _round_pool()
    dup = _extra(9, question=pool[0]["question"].upper())
    with pytest.warns(UserWarning, match="extra-9"):
        result = _round(extra_rows=[_extra(1), dup])
    assert result["dropped_duplicates"] == ["extra-9"]
    assert "extra-9" not in _ids_of(result)


def test_duplicates_among_the_extras_keep_the_first():
    twin = _extra(7, question=_extra(1)["question"])
    with pytest.warns(UserWarning, match="extra-7"):
        result = _round(extra_rows=[_extra(1), twin])
    assert result["dropped_duplicates"] == ["extra-7"]
    assert result["new"] == 1


def test_extras_that_are_all_duplicates_are_rejected():
    pool = _round_pool()
    with pytest.raises(ValueError, match="duplicate"), pytest.warns(UserWarning):
        _round(extra_rows=[_extra(9, question=pool[0]["question"])])


def test_an_extra_that_leaks_a_test_question_stops_the_round_naming_it():
    leaking = _extra(5, question="hand written QUESTION number 1")
    with pytest.raises(ValueError, match="extra-5"):
        _round(extra_rows=[_extra(1), leaking], test_rows=[_trow(1)])


def test_an_extra_in_a_category_that_is_not_weak_is_kept_with_a_warning():
    with pytest.warns(UserWarning, match="extra-2"):
        result = _round(extra_rows=[_extra(1), _extra(2, category="misconception")])
    assert "extra-2" in _ids_of(result)


def test_the_round_is_seeded_and_shuffled_not_new_then_replay():
    first = _ids_of(_round(seed=3))
    assert first == _ids_of(_round(seed=3))
    assert first != _ids_of(_round(seed=4))
    grouped = [i for i in first if i.startswith("extra-")] + [
        i for i in first if not i.startswith("extra-")
    ]
    orders = [_ids_of(_round(seed=s)) for s in range(6)]
    assert any(
        order
        != [i for i in order if i.startswith("extra-")]
        + [i for i in order if not i.startswith("extra-")]
        for order in orders
    )
    assert sorted(first) == sorted(grouped)


def test_build_round_dataset_does_not_mutate_its_inputs():
    pool, extras = _round_pool(), [_extra(1), _extra(2)]
    before = ([dict(r) for r in pool], [dict(r) for r in extras])
    _round(train_rows=pool, extra_rows=extras)
    assert (pool, extras) == before


def test_the_round_summary_counts_rows_by_category():
    result = _round(replay_fraction=1.0)
    assert sum(result["by_category"].values()) == len(result["rows"])
    assert result["by_category"]["pun_ambiguity"] == 4


# ---- M9: load_extra_rows ----
def test_load_extra_rows_reads_and_validates_the_file(tmp_path):
    path = tmp_path / "extra.jsonl"
    no_source = {k: v for k, v in _extra(1).items() if k != "source"}
    data.write_jsonl(path, [no_source, _extra(2)])
    rows = data.load_extra_rows(path)
    assert [r["id"] for r in rows] == ["extra-1", "extra-2"]
    assert rows[0]["source"] == "user_written"


def test_load_extra_rows_with_a_missing_file_raises_naming_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        data.load_extra_rows(tmp_path / "nope.jsonl")


def test_load_extra_rows_with_an_empty_file_raises(tmp_path):
    path = tmp_path / "extra.jsonl"
    data.write_jsonl(path, [])
    with pytest.raises(ValueError, match="no rows"):
        data.load_extra_rows(path)


def test_load_extra_rows_rejects_a_row_without_a_chain_naming_it(tmp_path):
    path = tmp_path / "extra.jsonl"
    data.write_jsonl(path, [{k: v for k, v in _extra(4).items() if k != "chain"}])
    with pytest.raises(ValueError, match="extra-4"):
        data.load_extra_rows(path)


def test_load_extra_rows_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "extra.jsonl"
    data.write_jsonl(
        path, [_extra(1), _extra(1, question="Another question about something else?")]
    )
    with pytest.raises(ValueError, match="duplicate"):
        data.load_extra_rows(path)
