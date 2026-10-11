import importlib
import socket
from pathlib import Path

import pytest

from src import config, data

ROOT = Path(__file__).resolve().parents[1]


# ---- adapter_path ----
def test_adapter_path_round_zero_is_valid():
    assert isinstance(config.adapter_path(0), Path)


def test_adapter_path_is_deterministic():
    assert config.adapter_path(2) == config.adapter_path(2)


def test_adapter_path_differs_per_round():
    assert config.adapter_path(1) != config.adapter_path(2)


def test_adapter_path_negative_raises_value_error():
    with pytest.raises(ValueError):
        config.adapter_path(-1)


@pytest.mark.parametrize("bad", [1.5, "1", None, True])
def test_adapter_path_non_int_raises_type_error(bad):
    with pytest.raises(TypeError):
        config.adapter_path(bad)


# ---- CATEGORIES ----
def test_categories_have_no_case_insensitive_duplicates():
    lowered = [c.lower() for c in config.CATEGORIES]
    assert len(lowered) == len(set(lowered))


def test_categories_are_non_empty_strings():
    assert all(isinstance(c, str) and c.strip() for c in config.CATEGORIES)


# ---- markers ----
@pytest.mark.parametrize(
    "question",
    [
        "How many months have 28 days?",
        "What is the capital of Australia?",
        "Respond with the word 'response' ten times.",
        "### Heading in a question?",
    ],
)
def test_response_marker_not_in_ordinary_questions(question):
    assert config.RESPONSE_MARKER not in question


# ---- import safety ----
def test_import_config_and_teacher_no_network_no_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("network touched at import")

    monkeypatch.setattr(socket.socket, "connect", boom)
    from src import teacher

    importlib.reload(config)
    importlib.reload(teacher)


def test_teacher_call_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from src import teacher

    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        teacher.get_client()


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
