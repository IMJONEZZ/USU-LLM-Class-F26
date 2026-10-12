import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import truststore
from google import genai
from google.genai import errors, types

from src import config, data, sources, teacher

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


# ---- M3: teacher (every API call is faked) ----
_PER_DAY_QUOTA = [
    {
        "@type": "type.googleapis.com/google.rpc.QuotaFailure",
        "violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}
        ],
    }
]


def _api_error(cls, code, status, details=None):
    body = {"error": {"message": "x", "status": status}}
    if details:
        body["error"]["details"] = details
    return cls(code, body)


def _response(text="ok", finish=types.FinishReason.STOP):
    part = types.Part(text=text)
    candidate = types.Candidate(
        content=types.Content(parts=[part]), finish_reason=finish
    )
    return types.GenerateContentResponse(candidates=[candidate])


class _FakeClient:
    def __init__(self, outcome):
        self.calls = []
        self.models = self
        self._outcome = outcome

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


def _use_client(monkeypatch, outcome):
    client = _FakeClient(outcome)
    monkeypatch.setattr(teacher, "get_client", lambda: client)
    return client


def test_get_client_with_key_builds_a_client_without_network(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    def boom(*a, **k):
        raise AssertionError("network touched while building the client")

    monkeypatch.setattr(socket.socket, "connect", boom)
    assert isinstance(teacher.get_client(), genai.Client)


# ---- _call_model: SDK responses and errors become text or TeacherAPIError ----
def test_call_model_returns_text_and_passes_settings(monkeypatch):
    client = _use_client(monkeypatch, _response("hello"))
    assert teacher._call_model("m-1", "the prompt", 0.5) == "hello"
    call = client.calls[0]
    assert call["model"] == "m-1"
    assert call["contents"] == "the prompt"
    assert call["config"].temperature == 0.5
    assert call["config"].max_output_tokens == config.TEACHER_MAX_OUTPUT_TOKENS


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_api_error(errors.ClientError, 401, "UNAUTHENTICATED"), "auth"),
        (_api_error(errors.ClientError, 403, "PERMISSION_DENIED"), "auth"),
        (_api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"), "rate_limit"),
        (
            _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", _PER_DAY_QUOTA),
            "quota",
        ),
        (_api_error(errors.ClientError, 400, "INVALID_ARGUMENT"), "bad_request"),
        (_api_error(errors.ServerError, 503, "UNAVAILABLE"), "server"),
        (httpx.ReadTimeout("slow"), "timeout"),
    ],
)
def test_call_model_maps_errors_to_kinds(monkeypatch, error, kind):
    _use_client(monkeypatch, error)
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == kind


@pytest.mark.parametrize(
    ("response", "kind"),
    [
        (_response(""), "empty"),
        (types.GenerateContentResponse(candidates=[]), "empty"),
        (_response("cut off", types.FinishReason.MAX_TOKENS), "truncated"),
        (_response("nope", types.FinishReason.SAFETY), "blocked"),
        (
            types.GenerateContentResponse(
                prompt_feedback=types.GenerateContentResponsePromptFeedback(
                    block_reason=types.BlockedReason.SAFETY
                )
            ),
            "blocked",
        ),
    ],
)
def test_call_model_rejects_unusable_responses(monkeypatch, response, kind):
    _use_client(monkeypatch, response)
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == kind


def test_call_model_missing_key_message_is_clear(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        teacher._call_model("m", "p", 0.0)


# ---- _generate: retry only the transient failures ----
def _script_call_model(monkeypatch, outcomes):
    calls, sleeps = [], []
    queue = list(outcomes)

    def fake(model, prompt, temperature):
        calls.append(prompt)
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_call_model", fake)
    monkeypatch.setattr(teacher.time, "sleep", sleeps.append)
    return calls, sleeps


def test_generate_retries_transient_errors_then_succeeds(monkeypatch):
    calls, sleeps = _script_call_model(
        monkeypatch,
        [
            teacher.TeacherAPIError("rate_limit"),
            teacher.TeacherAPIError("timeout"),
            "fine",
        ],
    )
    assert teacher._generate("p", "m", 0.0) == "fine"
    assert len(calls) == 3
    assert sleeps == [config.TEACHER_RETRY_SECONDS] * 2


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "server"])
def test_generate_gives_up_after_max_retries(monkeypatch, kind):
    attempts = 1 + config.TEACHER_MAX_RETRIES
    calls, _ = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError(kind)] * attempts
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == attempts


@pytest.mark.parametrize(
    "kind", ["auth", "quota", "bad_request", "empty", "blocked", "truncated"]
)
def test_generate_does_not_retry_permanent_errors(monkeypatch, kind):
    calls, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError(kind), "never reached"]
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.kind == kind
    assert len(calls) == 1
    assert sleeps == []


# ---- chain_prompt ----
def test_chain_prompt_has_question_key_and_all_section_labels():
    prompt = teacher.chain_prompt("How many sheep are left?", "zebra-77")
    assert "How many sheep are left?" in prompt
    assert "zebra-77" in prompt
    assert all(label in prompt for label in config.SECTION_LABELS)


def test_chain_prompt_short_key_must_be_the_final_answer_verbatim():
    prompt = teacher.chain_prompt("Q?", "zebra-77")
    assert f"{config.FINAL_ANSWER_LABEL} zebra-77" in prompt


def test_chain_prompt_long_key_asks_for_a_short_final_answer():
    key = "A long rebuttal sentence. " * 5
    assert len(key) > config.LONG_KEY_CHARS
    prompt = teacher.chain_prompt("Q?", key)
    assert key in prompt
    assert f"{config.FINAL_ANSWER_LABEL} {key}" not in prompt
    assert "short" in prompt


# ---- generate_chain ----
ROW = {
    "id": "x-1",
    "question": "How many sheep are left?",
    "answer": "9",
    "accepted_alternatives": ["nine"],
    "category": "lateral_thinking",
    "source": "brainteaser",
}


def _teacher_chain(final="9", filler=""):
    return (
        "Restate: A farmer has 17 sheep; all but 9 run away.\n"
        "Check: This is a wording trap, not subtraction.\n"
        f"Solve: All but 9 means 9 remain. {filler}\n"
        "Verify: 17 - 8 = 9, so 9 stay.\n"
        f"Final answer: {final}"
    )


def _script_generate(monkeypatch, outcomes):
    prompts, models = [], []
    queue = list(outcomes)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        models.append(model)
        outcome = queue.pop(0) if queue else teacher.TeacherAPIError("empty")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, models


def test_generate_chain_adds_only_a_chain_to_the_row(monkeypatch):
    chain = _teacher_chain("9")
    prompts, models = _script_generate(monkeypatch, [chain])
    assert teacher.generate_chain(ROW) == {**ROW, "chain": chain}
    assert len(prompts) == 1
    assert models == [config.TEACHER_MODEL]


def test_generate_chain_prompt_has_question_and_the_correct_answer(monkeypatch):
    row = {**ROW, "answer": "zebra-77"}
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("zebra-77")])
    teacher.generate_chain(row)
    assert "How many sheep are left?" in prompts[0]
    assert "zebra-77" in prompts[0]


def test_generate_chain_retries_until_a_chain_is_valid(monkeypatch):
    good = _teacher_chain("9")
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("12"), good])
    assert teacher.generate_chain(ROW)["chain"] == good
    assert len(prompts) == 2


def test_generate_chain_returns_none_after_all_attempts_fail(monkeypatch):
    prompts, _ = _script_generate(monkeypatch, [_teacher_chain("12")] * 20)
    assert teacher.generate_chain(ROW) is None
    assert len(prompts) == config.TEACHER_ATTEMPTS


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated"])
def test_generate_chain_skips_content_level_failures(monkeypatch, kind):
    good = _teacher_chain("9")
    _script_generate(monkeypatch, [teacher.TeacherAPIError(kind), good])
    assert teacher.generate_chain(ROW)["chain"] == good


@pytest.mark.parametrize(
    "kind", ["auth", "quota", "rate_limit", "timeout", "server", "bad_request"]
)
def test_generate_chain_stops_on_errors_that_survive_the_retries(monkeypatch, kind):
    prompts, _ = _script_generate(
        monkeypatch, [teacher.TeacherAPIError(kind, "why"), _teacher_chain("9")]
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher.generate_chain(ROW)
    assert caught.value.kind == kind
    assert len(prompts) == 1


def test_generate_chain_accepts_alternative_answers(monkeypatch):
    _script_generate(monkeypatch, [_teacher_chain("nine")])
    assert teacher.generate_chain(ROW) is not None


@pytest.mark.parametrize(
    "wrapper", ["```\n{c}\n```", "```text\n{c}\n```", "  \n{c}\n\n"]
)
def test_generate_chain_unwraps_code_fences_and_whitespace(monkeypatch, wrapper):
    chain = _teacher_chain("9")
    _script_generate(monkeypatch, [wrapper.format(c=chain)])
    assert teacher.generate_chain(ROW)["chain"] == chain


def _without_line(chain, label):
    return "\n".join(l for l in chain.split("\n") if not l.startswith(label))


def _swap_lines(chain, i, j):
    lines = chain.split("\n")
    lines[i], lines[j] = lines[j], lines[i]
    return "\n".join(lines)


@pytest.mark.parametrize(
    "bad",
    [
        "The answer is 9.",
        _without_line(_teacher_chain("9"), "Final answer:") + "\nAnswer: 9",
        _swap_lines(_teacher_chain("9"), 1, 2),
        "Restate: a\nCheck: b\nSolve: c\nVerify: d\nFinal answer: 9",
        _teacher_chain("9", "x" * 2000),
        _teacher_chain("9", "### Response:"),
    ],
    ids=[
        "no-labels",
        "no-final-label",
        "wrong-order",
        "too-short",
        "too-long",
        "marker",
    ],
)
def test_generate_chain_rejects_malformed_chains(monkeypatch, bad):
    _script_generate(monkeypatch, [bad] * 20)
    assert teacher.generate_chain(ROW) is None


@pytest.mark.parametrize(
    ("key", "final", "accepted"),
    [
        ("this is impossible", "This is impossible.", True),
        ("5 km", "5 km", True),
        ("5 km", "5", False),
        ("5", "5 km", False),
        ("yes", "Yes, because it is", False),
    ],
)
def test_generate_chain_short_keys_must_match_the_final_answer(
    monkeypatch, key, final, accepted
):
    _script_generate(monkeypatch, [_teacher_chain(final)] * 20)
    row = {**ROW, "answer": key, "accepted_alternatives": []}
    assert (teacher.generate_chain(row) is not None) is accepted


def test_generate_chain_long_keys_accept_any_valid_chain(monkeypatch):
    key = "A long rebuttal that no model would repeat word for word. " * 2
    assert len(key) > config.LONG_KEY_CHARS
    chain = _teacher_chain("A short version")
    _script_generate(monkeypatch, [chain])
    row = {**ROW, "answer": key, "accepted_alternatives": []}
    assert teacher.generate_chain(row)["chain"] == chain


def test_generate_chain_does_not_mutate_the_row_and_result_is_valid(monkeypatch):
    before = dict(ROW)
    _script_generate(monkeypatch, [_teacher_chain("9")])
    result = teacher.generate_chain(ROW)
    assert ROW == before
    data.check_schema(result)


# ---- build_training_data (generate_chain is faked) ----
def _train_rows(n=4):
    return [
        {
            "id": f"r-{i}",
            "question": f"Question number {i}?",
            "answer": "9",
            "accepted_alternatives": ["nine"],
            "category": config.CATEGORIES[i % len(config.CATEGORIES)],
            "source": "brainteaser",
        }
        for i in range(n)
    ]


def _fake_generate_chain(monkeypatch, fail_ids=(), crash_on=None, crash_with=None):
    seen = []

    def fake(row):
        seen.append(row)
        if row["id"] == crash_on:
            raise crash_with or RuntimeError("boom")
        if row["id"] in fail_ids:
            return None
        return {**row, "chain": "CHAIN-" + row["id"]}

    monkeypatch.setattr(teacher, "generate_chain", fake)
    return seen


def _paths(tmp_path):
    return tmp_path / "train.jsonl", tmp_path / "train.partial.jsonl"


def test_build_training_data_writes_rows_in_input_order(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    written = data.read_jsonl(out)
    assert [r["id"] for r in written] == ["r-0", "r-1", "r-2", "r-3"]
    assert all("chain" in r for r in written)
    assert summary["written"] == 4
    assert not partial.exists()


def test_build_training_data_refuses_existing_output_without_force(
    tmp_path, monkeypatch
):
    seen = _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        teacher.build_training_data(_train_rows(2), out, partial)
    assert seen == []
    assert out.read_text(encoding="utf-8") == "old\n"


def test_build_training_data_force_overwrites(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    teacher.build_training_data(_train_rows(2), out, partial, force=True)
    assert [r["id"] for r in data.read_jsonl(out)] == ["r-0", "r-1"]


def test_build_training_data_resumes_after_a_crash(tmp_path, monkeypatch):
    out, partial = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, crash_on="r-2")
    with pytest.raises(RuntimeError, match="boom"):
        teacher.build_training_data(_train_rows(4), out, partial)
    assert [r["id"] for r in data.read_jsonl(partial)] == ["r-0", "r-1"]
    assert not out.exists()

    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    assert [r["id"] for r in seen] == ["r-2", "r-3"]
    assert [r["id"] for r in data.read_jsonl(out)] == ["r-0", "r-1", "r-2", "r-3"]
    assert summary["resumed"] == 2
    assert not partial.exists()


def test_build_training_data_ignores_stale_partial_rows(tmp_path, monkeypatch):
    out, partial = _paths(tmp_path)
    keep = {**_train_rows(1)[0], "chain": "CHAIN-r-0"}
    stale = {**keep, "id": "stale-9"}
    data.write_jsonl(partial, [keep, stale])
    seen = _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, partial)
    assert [r["id"] for r in seen] == ["r-1"]
    assert [r["id"] for r in data.read_jsonl(out)] == ["r-0", "r-1"]


def test_build_training_data_skips_rows_without_a_valid_chain(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    assert [r["id"] for r in data.read_jsonl(out)] == ["r-0", "r-2", "r-3"]
    assert summary["skipped"] == [{"id": "r-1", "reason": "no_valid_chain"}]


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit"])
def test_build_training_data_api_error_stops_and_keeps_earlier_rows(
    tmp_path, monkeypatch, kind
):
    out, partial = _paths(tmp_path)
    _fake_generate_chain(
        monkeypatch, crash_on="r-2", crash_with=teacher.TeacherAPIError(kind, "why")
    )
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.build_training_data(_train_rows(4), out, partial)
    assert [r["id"] for r in data.read_jsonl(partial)] == ["r-0", "r-1"]
    assert not out.exists()


def test_build_training_data_empty_input_raises_and_writes_nothing(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        teacher.build_training_data([], out, partial)
    assert not out.exists()
    assert not partial.exists()


def test_build_training_data_duplicate_ids_raise(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    rows = _train_rows(2)
    rows[1] = {**rows[1], "id": rows[0]["id"]}
    with pytest.raises(ValueError, match="id"):
        teacher.build_training_data(rows, out, partial)


def test_build_training_data_progress_is_called_once_per_input_row(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    out, partial = _paths(tmp_path)
    ticks = []
    teacher.build_training_data(
        _train_rows(4), out, partial, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 4


def test_build_training_data_summary_counts(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch, fail_ids={"r-3"})
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    assert summary["written"] == 3
    assert summary["resumed"] == 0
    assert summary["skipped"] == [{"id": "r-3", "reason": "no_valid_chain"}]


# ---- timing of API calls ----
def _fake_clock(monkeypatch, step):
    state = {"t": -step}

    def now():
        state["t"] += step
        return state["t"]

    monkeypatch.setattr(teacher, "_now", now)


def test_generate_logs_one_record_per_attempt_with_outcome_and_seconds(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, [teacher.TeacherAPIError("rate_limit"), "fine"])
    _fake_clock(monkeypatch, 2.5)
    assert teacher._generate("p", "m-1", 0.0) == "fine"
    assert teacher.call_log() == [
        {"model": "m-1", "seconds": 2.5, "outcome": "rate_limit"},
        {"model": "m-1", "seconds": 2.5, "outcome": "ok"},
    ]


def test_generate_logs_unexpected_exceptions_and_still_raises(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, [RuntimeError("weird")])
    _fake_clock(monkeypatch, 1.0)
    with pytest.raises(RuntimeError, match="weird"):
        teacher._generate("p", "m-1", 0.0)
    assert teacher.call_log() == [
        {"model": "m-1", "seconds": 1.0, "outcome": "RuntimeError"}
    ]


def test_call_log_returns_a_copy_and_reset_clears_it(monkeypatch):
    teacher.reset_call_log()
    _script_call_model(monkeypatch, ["fine"])
    teacher._generate("p", "m", 0.0)
    teacher.call_log().clear()
    assert len(teacher.call_log()) == 1
    teacher.reset_call_log()
    assert teacher.call_log() == []


def test_timing_summary_groups_by_model():
    records = [
        {"model": "pro", "seconds": 10.0, "outcome": "ok"},
        {"model": "pro", "seconds": 30.0, "outcome": "timeout"},
        {"model": "flash", "seconds": 1.0, "outcome": "ok"},
    ]
    assert teacher.timing_summary(records) == {
        "pro": {
            "calls": 2,
            "total_seconds": 40.0,
            "mean_seconds": 20.0,
            "max_seconds": 30.0,
            "outcomes": {"ok": 1, "timeout": 1},
        },
        "flash": {
            "calls": 1,
            "total_seconds": 1.0,
            "mean_seconds": 1.0,
            "max_seconds": 1.0,
            "outcomes": {"ok": 1},
        },
    }


def test_timing_summary_of_nothing_is_empty():
    assert teacher.timing_summary([]) == {}


def test_build_training_data_summary_has_row_seconds(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    assert summary["row_seconds"] == {"n": 4, "mean": 1.0, "max": 1.0}


def test_build_training_data_row_seconds_skip_resumed_rows(tmp_path, monkeypatch):
    out, partial = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, crash_on="r-2")
    with pytest.raises(RuntimeError):
        teacher.build_training_data(_train_rows(4), out, partial)
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    summary = teacher.build_training_data(_train_rows(4), out, partial)
    assert summary["row_seconds"]["n"] == 2


def test_build_training_data_row_seconds_when_nothing_was_generated(
    tmp_path, monkeypatch
):
    out, partial = _paths(tmp_path)
    rows = _train_rows(2)
    data.write_jsonl(partial, [{**r, "chain": "c"} for r in rows])
    _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(rows, out, partial)
    assert summary["row_seconds"] == {"n": 0, "mean": None, "max": None}


def test_build_training_data_timing_covers_only_this_run(tmp_path, monkeypatch):
    _script_call_model(monkeypatch, ["old call"])
    teacher._generate("p", "m", 0.0)
    _fake_generate_chain(monkeypatch)
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(2), out, partial)
    assert summary["timing"] == {}


def test_build_training_data_timing_reports_the_api_calls_it_made(
    tmp_path, monkeypatch
):
    _script_call_model(monkeypatch, ["fine", "fine"])

    def fake_generate_chain(row):
        teacher._generate("p", "m-1", 0.0)
        return {**row, "chain": "c"}

    monkeypatch.setattr(teacher, "generate_chain", fake_generate_chain)
    out, partial = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(2), out, partial)
    assert summary["timing"]["m-1"]["calls"] == 2
    assert summary["timing"]["m-1"]["outcomes"] == {"ok": 2}


# ---- scripts/pipeline.py (no network: these paths exit before any API call) ----
PIPELINE = str(ROOT / "scripts" / "pipeline.py")


def _run_pipeline(*args):
    return subprocess.run(
        [sys.executable, PIPELINE, *args], capture_output=True, text=True, check=False
    )


def test_pipeline_train_data_requires_per_category(tmp_path):
    result = _run_pipeline("train-data", "--out", str(tmp_path / "t.jsonl"))
    assert result.returncode != 0
    assert "per-category" in result.stderr


def test_pipeline_train_data_refuses_existing_output_before_any_network(tmp_path):
    out = tmp_path / "t.jsonl"
    out.write_text("old\n", encoding="utf-8")
    result = _run_pipeline("train-data", "--per-category", "1", "--out", str(out))
    assert result.returncode != 0
    assert "--force" in result.stderr
    assert out.read_text(encoding="utf-8") == "old\n"


def _chain_rows():
    rows = []
    for category in config.CATEGORIES:
        for i in range(3):
            rows.append(
                {**ROW, "id": f"{category}-{i}", "category": category, "chain": "c"}
            )
    return rows


def test_pipeline_sample_takes_per_category_rows_and_is_seeded(tmp_path):
    train = tmp_path / "train.jsonl"
    data.write_jsonl(train, _chain_rows())
    outs = []
    for name in ("a.json", "b.json"):
        out = tmp_path / name
        result = _run_pipeline(
            "sample",
            "--train",
            str(train),
            "--per-category",
            "2",
            "--seed",
            "0",
            "--out",
            str(out),
        )
        assert result.returncode == 0, result.stderr
        outs.append(out.read_text(encoding="utf-8"))
    picked = json.loads(outs[0])
    assert len(picked) == 2 * len(config.CATEGORIES)
    assert {r["category"] for r in picked} == set(config.CATEGORIES)
    assert outs[0] == outs[1]


def test_pipeline_sample_missing_train_file_fails_with_a_clear_message(tmp_path):
    result = _run_pipeline(
        "sample",
        "--train",
        str(tmp_path / "nope.jsonl"),
        "--per-category",
        "2",
        "--out",
        str(tmp_path / "o.json"),
    )
    assert result.returncode != 0
    assert "train-data" in result.stderr


# ---- M4: test set, leakage and hash ----
def _trow(i=1, **overrides):
    return {
        "id": f"t-{i}",
        "question": f"Hand written question number {i}?",
        "answer": "9",
        "accepted_alternatives": ["nine"],
        "category": "lateral_thinking",
        "source": "hand_written",
        "chain": _teacher_chain("9", f"variant {i}"),
        **overrides,
    }


def _train_row(i=1, **overrides):
    return {
        "id": f"tr-{i}",
        "question": f"Public training question {i}?",
        "answer": "a",
        "accepted_alternatives": [],
        "category": "misconception",
        "source": "truthfulqa",
        "chain": _teacher_chain("a", f"train {i}"),
        **overrides,
    }


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


# ---- load_test_set ----
def _write_test_set(tmp_path, rows):
    path = tmp_path / "test_set.jsonl"
    data.write_jsonl(path, rows)
    return path


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


# ---- scripts/pipeline.py: test-data and validate ----
def _write_all(tmp_path, test_rows, train_rows, eval_rows=None):
    test_path = tmp_path / "test_set.jsonl"
    train_path = tmp_path / "train.jsonl"
    data.write_jsonl(test_path, test_rows)
    data.write_jsonl(train_path, train_rows)
    if eval_rows is not None:
        data.write_jsonl(tmp_path / "eval.jsonl", eval_rows)
    return test_path, train_path, tmp_path / "eval.jsonl", tmp_path / "hash.jsonl"


def test_pipeline_test_data_records_the_hash(tmp_path):
    test_path, _, _, hash_path = _write_all(tmp_path, [_trow(1), _trow(2)], [])
    result = _run_pipeline(
        "test-data", "--test-set", str(test_path), "--hash", str(hash_path)
    )
    assert result.returncode == 0, result.stderr
    (record,) = data.read_jsonl(hash_path)
    assert record["sha256"] == data.file_sha256(test_path)
    assert "2 rows" in result.stdout


def test_pipeline_test_data_failure_names_the_row_and_records_nothing(tmp_path):
    rows = [_trow(1), _trow(2, chain=_teacher_chain("12"))]
    test_path, _, _, hash_path = _write_all(tmp_path, rows, [])
    result = _run_pipeline(
        "test-data", "--test-set", str(test_path), "--hash", str(hash_path)
    )
    assert result.returncode != 0
    assert "t-2" in result.stderr
    assert not hash_path.exists()


def test_pipeline_validate_passes_on_clean_data(tmp_path):
    paths = _write_all(tmp_path, [_trow(1)], [_train_row(1)], eval_rows=[])
    test_path, train_path, eval_path, hash_path = paths
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr


def test_pipeline_validate_fails_on_leakage_naming_the_test_row(tmp_path):
    train = [_train_row(1, question="Hand written question number 1?")]
    test_path, train_path, eval_path, hash_path = _write_all(
        tmp_path, [_trow(1)], train
    )
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode != 0
    assert "t-1" in result.stderr


def test_pipeline_validate_warns_when_the_test_set_changed(tmp_path):
    test_path, train_path, eval_path, hash_path = _write_all(
        tmp_path, [_trow(1)], [_train_row(1)]
    )
    data.record_test_set_hash(test_path, hash_path)
    data.write_jsonl(test_path, [_trow(1), _trow(2)])
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert "changed" in result.stderr


def test_pipeline_validate_missing_train_file_fails_clearly(tmp_path):
    test_path, _, eval_path, hash_path = _write_all(tmp_path, [_trow(1)], [])
    result = _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(tmp_path / "nope.jsonl"),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
    )  # fmt: skip
    assert result.returncode != 0
    assert "train-data" in result.stderr


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
