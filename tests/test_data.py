import importlib
import socket
import subprocess
from pathlib import Path

import pytest
import truststore

from src import config, data, sources

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


# ---- M2b: sources (TruthfulQA, FalseQA, BrainTeaser, FreshQA) ----
TQA = {
    "type": "Adversarial",
    "category": "Misconceptions",
    "question": "What happens if you eat watermelon seeds?",
    "best_answer": "The watermelon seeds pass through your digestive system",
    "correct_answers": [
        "Nothing happens",
        "The watermelon seeds pass through your digestive system",
        "You eat watermelon seeds",
    ],
    "incorrect_answers": ["You grow watermelons in your stomach", "You get sick"],
    "source": "https://example.org",
}
TQA_REQUIRED = [
    "question",
    "category",
    "best_answer",
    "correct_answers",
    "incorrect_answers",
]


def _tqa(**overrides):
    return {**TQA, **overrides}


def test_parse_truthfulqa_builds_schema_row():
    (row,) = sources.parse_truthfulqa([TQA])
    data.check_schema(row)
    assert row["id"] == "tqa-0000"
    assert row["source"] == "truthfulqa"
    assert row["category"] == "misconception"
    assert row["question"] == TQA["question"]
    assert row["answer"] == TQA["best_answer"]


def test_parse_truthfulqa_alternatives_are_other_correct_answers():
    (row,) = sources.parse_truthfulqa([TQA])
    assert row["accepted_alternatives"] == [
        "Nothing happens",
        "You eat watermelon seeds",
    ]


def test_parse_truthfulqa_incorrect_answers_never_become_alternatives():
    record = _tqa(incorrect_answers=["Nothing happens", "You get sick"])
    (row,) = sources.parse_truthfulqa([record])
    assert "Nothing happens" not in row["accepted_alternatives"]
    assert "You get sick" not in row["accepted_alternatives"]


@pytest.mark.parametrize(
    ("source_category", "expected"),
    [
        ("Misconceptions", "misconception"),
        ("Misconceptions: Topical", "misconception"),
        ("Superstitions", "misconception"),
        ("Myths and Fairytales", "misconception"),
        ("Mandela Effect", "misconception"),
        ("Indexical Error: Time", "time_sensitive"),
    ],
)
def test_parse_truthfulqa_category_mapping(source_category, expected):
    rows = sources.parse_truthfulqa([_tqa(category=source_category)])
    assert [r["category"] for r in rows] == [expected]


def test_parse_truthfulqa_drops_unmapped_category():
    assert sources.parse_truthfulqa([_tqa(category="Law")]) == []


def test_parse_truthfulqa_ids_follow_source_position_not_output_position():
    rows = sources.parse_truthfulqa([_tqa(category="Law"), _tqa(question="Second?")])
    assert [r["id"] for r in rows] == ["tqa-0001"]


def test_parse_truthfulqa_strips_whitespace():
    (row,) = sources.parse_truthfulqa([_tqa(question="  Q?  ", best_answer=" A ")])
    assert (row["question"], row["answer"]) == ("Q?", "A")


def test_parse_truthfulqa_empty_input_is_empty_list():
    assert sources.parse_truthfulqa([]) == []


@pytest.mark.parametrize("column", TQA_REQUIRED)
def test_parse_truthfulqa_missing_column_names_it(column):
    record = {k: v for k, v in TQA.items() if k != column}
    with pytest.raises(ValueError, match=column):
        sources.parse_truthfulqa([record])


@pytest.mark.parametrize("field", ["question", "best_answer"])
def test_parse_truthfulqa_blank_text_raises(field):
    with pytest.raises(ValueError, match=field):
        sources.parse_truthfulqa([_tqa(**{field: "   "})])


def test_parse_truthfulqa_correct_answers_must_be_a_list():
    with pytest.raises(ValueError, match="correct_answers"):
        sources.parse_truthfulqa([_tqa(correct_answers="Nothing happens")])


# ---- FalseQA ----
def _fq(**overrides):
    return {
        "question": "Why is CO2 made of oxygen?",
        "answer": "It has one carbon.",
        "label": "1",
        **overrides,
    }


def test_parse_falseqa_builds_schema_row():
    (row,) = sources.parse_falseqa([_fq()], "train")
    data.check_schema(row)
    assert row["id"] == "falseqa-train-0000"
    assert row["source"] == "falseqa"
    assert row["category"] == "false_premise"
    assert row["answer"] == "It has one carbon."
    assert row["accepted_alternatives"] == []


def test_parse_falseqa_stringified_list_answer_splits_into_answer_and_alternatives():
    (row,) = sources.parse_falseqa(
        [_fq(answer="['First.', 'Second.', 'Third.']")], "test"
    )
    assert row["answer"] == "First."
    assert row["accepted_alternatives"] == ["Second.", "Third."]


def test_parse_falseqa_drops_true_premise_rows():
    assert sources.parse_falseqa([_fq(label="0")], "train") == []


def test_parse_falseqa_unknown_label_raises():
    with pytest.raises(ValueError, match="label"):
        sources.parse_falseqa([_fq(label="2")], "train")


@pytest.mark.parametrize("column", ["question", "answer", "label"])
def test_parse_falseqa_missing_column_names_it(column):
    record = {k: v for k, v in _fq().items() if k != column}
    with pytest.raises(ValueError, match=column):
        sources.parse_falseqa([record], "train")


@pytest.mark.parametrize(
    "override", [{"question": " "}, {"answer": ""}, {"answer": "[]"}]
)
def test_parse_falseqa_blank_question_or_answer_raises(override):
    with pytest.raises(ValueError):
        sources.parse_falseqa([_fq(**override)], "train")


def test_parse_falseqa_malformed_list_answer_raises():
    with pytest.raises(ValueError, match="answer"):
        sources.parse_falseqa([_fq(answer="['unterminated")], "train")


def test_parse_falseqa_empty_input_is_empty_list():
    assert sources.parse_falseqa([], "train") == []


# ---- BrainTeaser (reads the hand-converted JSONL) ----
BT = {
    "id": "SP-0",
    "question": "Mr. and Mrs. Mustard have six daughters and each has one brother. Why only 9 people?",
    "answer": "Each daughter shares the same brother.",
    "label": 2,
    "choice_list": [
        "Some brothers moved away.",
        "Some daughters got married.",
        "Each daughter shares the same brother.",
        "None of above.",
    ],
    "kind": "sentence",
}


def _bt(**overrides):
    return {**BT, **overrides}


def test_parse_brainteaser_sentence_puzzle_is_lateral_thinking():
    (row,) = sources.parse_brainteaser([BT])
    data.check_schema(row)
    assert row["id"] == "brainteaser-SP-0"
    assert row["source"] == "brainteaser"
    assert row["category"] == "lateral_thinking"
    assert row["answer"] == BT["answer"]
    assert row["accepted_alternatives"] == []


def test_parse_brainteaser_word_puzzle_is_pun_ambiguity():
    word = _bt(id="WP-0", kind="word", answer="Some daughters got married.", label=1)
    (row,) = sources.parse_brainteaser([word])
    assert row["category"] == "pun_ambiguity"


@pytest.mark.parametrize("suffix", ["_SR", "_CR"])
def test_parse_brainteaser_drops_reconstruction_variants(suffix):
    assert sources.parse_brainteaser([_bt(id="SP-0" + suffix)]) == []


@pytest.mark.parametrize("label", [4, -1, "2", None, 2.0])
def test_parse_brainteaser_bad_label_raises(label):
    with pytest.raises(ValueError, match="label"):
        sources.parse_brainteaser([_bt(label=label)])


def test_parse_brainteaser_answer_not_matching_choice_is_dropped_with_warning():
    with pytest.warns(UserWarning, match="SP-1"):
        rows = sources.parse_brainteaser([_bt(id="SP-1", label=0)])
    assert rows == []


def test_parse_brainteaser_duplicate_choices_dropped_with_warning():
    dup = _bt(
        id="SP-2", choice_list=["a", "a", "b", "None of above."], label=0, answer="a"
    )
    with pytest.warns(UserWarning, match="SP-2"):
        rows = sources.parse_brainteaser([dup])
    assert rows == []


def test_parse_brainteaser_none_of_above_answer_dropped_with_warning():
    none = _bt(id="SP-3", label=3, answer="None of above.")
    with pytest.warns(UserWarning, match="SP-3"):
        rows = sources.parse_brainteaser([none])
    assert rows == []


def test_parse_brainteaser_unknown_kind_raises():
    with pytest.raises(ValueError, match="kind"):
        sources.parse_brainteaser([_bt(kind="riddle")])


@pytest.mark.parametrize(
    "column", ["id", "question", "answer", "label", "choice_list", "kind"]
)
def test_parse_brainteaser_missing_column_names_it(column):
    record = {k: v for k, v in BT.items() if k != column}
    with pytest.raises(ValueError, match=column):
        sources.parse_brainteaser([record])


def test_parse_brainteaser_blank_question_raises():
    with pytest.raises(ValueError, match="question"):
        sources.parse_brainteaser([_bt(question="  ")])


# ---- FreshQA ----
_FRESH_COLUMNS = (
    ["id", "split", "question", "effective_year", "next_review", "false_premise"]
    + ["num_hops", "fact_type", "source"]
    + [f"answer_{i}" for i in range(10)]
    + ["note"]
)


def _fresh_csv(*rows, junk=True):
    lines = ["WARNING: do not click thumbs", ""] if junk else []
    lines.append(",".join(_FRESH_COLUMNS))
    for row in rows:
        lines.append(",".join(str(row.get(c, "")) for c in _FRESH_COLUMNS))
    return "\n".join(lines) + "\n"


def _fresh(**overrides):
    base = {
        "id": 1,
        "split": "TEST",
        "question": "Who runs it",
        "effective_year": 2026,
        "false_premise": "FALSE",
        "fact_type": "fast-changing",
        "answer_0": "Alice",
        "answer_1": "Alice Smith",
    }
    return {**base, **overrides}


def test_parse_freshqa_current_fast_changing_row_is_time_sensitive():
    (row,) = sources.parse_freshqa(_fresh_csv(_fresh()))
    data.check_schema(row)
    assert row["id"] == "freshqa-1"
    assert row["source"] == "freshqa"
    assert row["category"] == "time_sensitive"
    assert row["answer"] == "Alice"
    assert row["accepted_alternatives"] == ["Alice Smith"]


@pytest.mark.parametrize("year", [2025, 2022, "before 2022"])
@pytest.mark.parametrize("fact_type", ["fast-changing", "slow-changing"])
def test_parse_freshqa_drops_stale_changing_rows(year, fact_type):
    assert (
        sources.parse_freshqa(
            _fresh_csv(_fresh(effective_year=year, fact_type=fact_type))
        )
        == []
    )


def test_parse_freshqa_slow_changing_current_row_is_time_sensitive():
    (row,) = sources.parse_freshqa(_fresh_csv(_fresh(fact_type="slow-changing")))
    assert row["category"] == "time_sensitive"


def test_parse_freshqa_never_changing_non_false_premise_is_dropped():
    assert sources.parse_freshqa(_fresh_csv(_fresh(fact_type="never-changing"))) == []


def test_parse_freshqa_false_premise_never_changing_kept_any_year():
    csv_text = _fresh_csv(
        _fresh(
            false_premise="TRUE",
            fact_type="never-changing",
            effective_year="before 2022",
        )
    )
    (row,) = sources.parse_freshqa(csv_text)
    assert row["category"] == "false_premise"


def test_parse_freshqa_false_premise_stale_changing_dropped_current_kept():
    stale = _fresh(
        id=1, false_premise="TRUE", fact_type="slow-changing", effective_year=2024
    )
    current = _fresh(id=2, false_premise="TRUE", fact_type="slow-changing")
    rows = sources.parse_freshqa(_fresh_csv(stale, current))
    assert [r["id"] for r in rows] == ["freshqa-2"]


def test_parse_freshqa_uses_reference_year_at_call_time(monkeypatch):
    csv_text = _fresh_csv(
        _fresh(id=1, effective_year=2026), _fresh(id=2, effective_year=2027)
    )
    monkeypatch.setattr(config, "REFERENCE_YEAR", 2027)
    assert [r["id"] for r in sources.parse_freshqa(csv_text)] == ["freshqa-2"]


def test_parse_freshqa_finds_header_with_or_without_leading_junk():
    assert len(sources.parse_freshqa(_fresh_csv(_fresh(), junk=True))) == 1
    assert len(sources.parse_freshqa(_fresh_csv(_fresh(), junk=False))) == 1


def test_parse_freshqa_no_header_row_raises():
    with pytest.raises(ValueError, match="id"):
        sources.parse_freshqa("just,some,junk\n1,2,3\n")


@pytest.mark.parametrize("bad", ["true", "yes", ""])
def test_parse_freshqa_bad_false_premise_value_raises(bad):
    with pytest.raises(ValueError, match="false_premise"):
        sources.parse_freshqa(_fresh_csv(_fresh(false_premise=bad)))


def test_parse_freshqa_row_with_no_answers_raises():
    with pytest.raises(ValueError, match="answer"):
        sources.parse_freshqa(_fresh_csv(_fresh(answer_0="", answer_1="")))


def test_parse_freshqa_skips_empty_answer_cells_and_duplicates():
    row_in = _fresh(answer_0="Alice", answer_1="", answer_2="alice", answer_3="Al")
    (row,) = sources.parse_freshqa(_fresh_csv(row_in))
    assert row["accepted_alternatives"] == ["Al"]


def test_parse_freshqa_blank_question_raises():
    with pytest.raises(ValueError, match="question"):
        sources.parse_freshqa(_fresh_csv(_fresh(question="  ")))


def test_parse_freshqa_missing_column_raises():
    text = _fresh_csv(_fresh()).replace("fact_type", "fact_kind")
    with pytest.raises(ValueError, match="fact_type"):
        sources.parse_freshqa(text)


# ---- loaders (fetch is faked; nothing here touches the network) ----
def test_load_truthfulqa_uses_generation_config_and_parses(monkeypatch):
    calls = []
    monkeypatch.setattr(
        sources, "_load_hf", lambda name, cfg: calls.append((name, cfg)) or [TQA]
    )
    rows = sources.load_truthfulqa()
    assert calls == [("truthfulqa/truthful_qa", "generation")]
    assert [r["id"] for r in rows] == ["tqa-0000"]


def test_load_truthfulqa_network_failure_raises_not_empty(monkeypatch):
    def boom(name, cfg):
        raise ConnectionError("offline")

    monkeypatch.setattr(sources, "_load_hf", boom)
    with pytest.raises(ConnectionError):
        sources.load_truthfulqa()


def test_load_falseqa_fetches_every_split(monkeypatch):
    text = "question,answer,label\nWhy is CO2 made of oxygen?,It has one carbon.,1\n"
    monkeypatch.setattr(sources, "_fetch_text", lambda url: text)
    ids = [r["id"] for r in sources.load_falseqa()]
    assert ids == ["falseqa-train-0000", "falseqa-valid-0000", "falseqa-test-0000"]


def test_load_falseqa_network_failure_raises_not_empty(monkeypatch):
    def boom(url):
        raise OSError("offline")

    monkeypatch.setattr(sources, "_fetch_text", boom)
    with pytest.raises(OSError):
        sources.load_falseqa()


def test_load_falseqa_csv_with_wrong_columns_raises(monkeypatch):
    monkeypatch.setattr(sources, "_fetch_text", lambda url: "q,a,l\nx,y,1\n")
    with pytest.raises(ValueError, match="question"):
        sources.load_falseqa()


def test_load_brainteaser_reads_configured_jsonl(monkeypatch, tmp_path):
    path = tmp_path / "bt.jsonl"
    data.write_jsonl(path, [BT])
    monkeypatch.setattr(config, "BRAINTEASER_PATH", path)
    assert [r["id"] for r in sources.load_brainteaser()] == ["brainteaser-SP-0"]


def test_load_brainteaser_missing_file_raises_not_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BRAINTEASER_PATH", tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        sources.load_brainteaser()


def test_load_brainteaser_corrupt_jsonl_raises(monkeypatch, tmp_path):
    path = tmp_path / "bt.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    monkeypatch.setattr(config, "BRAINTEASER_PATH", path)
    with pytest.raises(ValueError, match="line 1"):
        sources.load_brainteaser()


def test_load_freshqa_fetches_and_parses(monkeypatch):
    monkeypatch.setattr(sources, "_fetch_text", lambda url: _fresh_csv(_fresh()))
    assert [r["id"] for r in sources.load_freshqa()] == ["freshqa-1"]


def test_load_freshqa_network_failure_raises_not_empty(monkeypatch):
    def boom(url):
        raise OSError("offline")

    monkeypatch.setattr(sources, "_fetch_text", boom)
    with pytest.raises(OSError):
        sources.load_freshqa()


def test_fetch_text_reads_utf8_from_a_url(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("Café,日本語\n", encoding="utf-8")
    assert sources._fetch_text(path.as_uri()) == "Café,日本語\n"


def test_fetch_text_unreachable_url_raises(tmp_path):
    with pytest.raises(OSError):
        sources._fetch_text((tmp_path / "nope.csv").as_uri())


def test_import_sources_makes_no_network_call(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network touched at import")

    monkeypatch.setattr(socket.socket, "connect", boom)
    importlib.reload(sources)


# ---- drop_duplicate_questions / load_all ----
def _row(i, question, category="misconception", source="truthfulqa"):
    return {
        "id": f"{source}-{i}",
        "question": question,
        "answer": "a",
        "accepted_alternatives": [],
        "category": category,
        "source": source,
    }


def test_drop_duplicate_questions_keeps_first_ignoring_case_and_punctuation():
    rows = [_row(1, "What is Nine?"), _row(2, "what is nine"), _row(3, "Other?")]
    with pytest.warns(UserWarning):
        kept = sources.drop_duplicate_questions(rows)
    assert [r["id"] for r in kept] == ["truthfulqa-1", "truthfulqa-3"]


def test_drop_duplicate_questions_no_duplicates_no_warning(recwarn):
    rows = [_row(1, "One?"), _row(2, "Two?")]
    assert sources.drop_duplicate_questions(rows) == rows
    assert len(recwarn) == 0


def test_drop_duplicate_questions_duplicate_ids_raise():
    rows = [_row(1, "One?"), _row(1, "Two?")]
    with pytest.raises(ValueError, match="id"):
        sources.drop_duplicate_questions(rows)


def _fake_loaders(monkeypatch, **overrides):
    rows = {
        "truthfulqa": [_row(1, "Q one?")],
        "falseqa": [_row(1, "Q two?", "false_premise", "falseqa")],
        "brainteaser": [_row(1, "Q three?", "lateral_thinking", "brainteaser")],
        "freshqa": [_row(1, "Q four?", "time_sensitive", "freshqa")],
        **overrides,
    }
    for name, value in rows.items():
        monkeypatch.setattr(sources, f"load_{name}", lambda value=value: value)


def test_load_all_concatenates_in_fixed_source_order(monkeypatch):
    _fake_loaders(monkeypatch)
    assert [r["source"] for r in sources.load_all()] == [
        "truthfulqa",
        "falseqa",
        "brainteaser",
        "freshqa",
    ]


def test_load_all_drops_cross_source_duplicate_keeping_earlier_source(monkeypatch):
    _fake_loaders(monkeypatch, falseqa=[_row(1, "q one", "false_premise", "falseqa")])
    with pytest.warns(UserWarning):
        rows = sources.load_all()
    assert "falseqa" not in {r["source"] for r in rows}


def test_load_all_empty_source_raises_naming_it(monkeypatch):
    _fake_loaders(monkeypatch, falseqa=[])
    with pytest.raises(ValueError, match="falseqa"):
        sources.load_all()


def test_load_all_schema_failure_stops(monkeypatch):
    bad = _row(1, "Bad cat?", category="not_a_category")
    _fake_loaders(monkeypatch, truthfulqa=[bad])
    with pytest.raises(ValueError, match="category"):
        sources.load_all()


def test_load_all_duplicate_ids_across_sources_raise(monkeypatch):
    clash = _row(1, "Other?", source="truthfulqa")
    _fake_loaders(monkeypatch, freshqa=[{**clash, "category": "time_sensitive"}])
    with pytest.raises(ValueError, match="id"):
        sources.load_all()


# ---- M2b: TLS behind a corporate proxy (Zscaler) ----
def test_fetch_text_uses_the_system_trust_store(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout, context):
        captured["context"] = context
        return FakeResponse()

    monkeypatch.setattr(sources.urllib.request, "urlopen", fake_urlopen)
    assert sources._fetch_text("https://example.org/x") == "ok"
    assert isinstance(captured["context"], truststore.SSLContext)


# ---- M2b follow-ups from the real-data sample ----
def _git_ignored(path):
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=ROOT, check=False
        ).returncode
        == 0
    )


def test_data_dir_is_ignored():
    assert _git_ignored("data/train.jsonl")


def test_assets_dir_is_ignored():
    # decision: no dataset is committed, including the hand-written ones
    assert _git_ignored("assets/test_set.jsonl")
    assert _git_ignored("assets/brainteaser.jsonl")


@pytest.mark.parametrize(
    "abstention", ["I have no comment", "No comment.", "I have no comment on that"]
)
def test_parse_truthfulqa_drops_rows_whose_best_answer_is_an_abstention(abstention):
    assert sources.parse_truthfulqa([_tqa(best_answer=abstention)]) == []


def test_parse_truthfulqa_strips_abstentions_from_alternatives():
    record = _tqa(correct_answers=[*TQA["correct_answers"], "I have no comment"])
    (row,) = sources.parse_truthfulqa([record])
    assert row["accepted_alternatives"] == [
        "Nothing happens",
        "You eat watermelon seeds",
    ]


def test_parse_truthfulqa_keeps_answers_that_only_contain_the_letters():
    (row,) = sources.parse_truthfulqa(
        [_tqa(best_answer="There is no commentary on it")]
    )
    assert row["answer"] == "There is no commentary on it"
