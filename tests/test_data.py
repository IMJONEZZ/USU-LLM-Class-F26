import importlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import truststore
from google import genai
from google.genai import errors, types
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from src import config, data, sources, teacher, train
from src import eval as evaluation

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


# ---- build_training_data: one permanent chain store, nothing is ever redone ----
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
    return tmp_path / "train.jsonl", tmp_path / "chains.jsonl"


def _ids(path):
    return [r["id"] for r in data.read_jsonl(path)]


def test_build_training_data_writes_rows_in_input_order_and_fills_the_store(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]
    assert _ids(store) == ["r-0", "r-1", "r-2", "r-3"]
    assert all("chain" in r for r in data.read_jsonl(store))
    assert (summary["written"], summary["generated"], summary["reused"]) == (4, 4, 0)


def test_a_second_run_makes_no_calls_and_gives_the_same_file(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    first = out.read_text(encoding="utf-8")
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert seen == []
    assert out.read_text(encoding="utf-8") == first
    assert (summary["generated"], summary["reused"]) == (0, 4)
    assert summary["timing"] == {}


def test_extending_with_more_rows_generates_only_the_new_ones(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(6), out, store)
    assert [r["id"] for r in seen] == ["r-4", "r-5"]
    assert _ids(out) == [f"r-{i}" for i in range(6)]
    assert (summary["generated"], summary["reused"]) == (2, 4)


def test_a_crash_keeps_finished_rows_and_the_rerun_only_does_the_rest(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, crash_on="r-2")
    with pytest.raises(RuntimeError, match="boom"):
        teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(store) == ["r-0", "r-1"]
    assert not out.exists()
    seen = _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    assert [r["id"] for r in seen] == ["r-2", "r-3"]
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit"])
def test_build_training_data_api_error_stops_and_keeps_earlier_rows(
    tmp_path, monkeypatch, kind
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(
        monkeypatch, crash_on="r-2", crash_with=teacher.TeacherAPIError(kind, "why")
    )
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(store) == ["r-0", "r-1"]
    assert not out.exists()


def test_a_row_that_failed_is_recorded_and_never_retried(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    first = teacher.build_training_data(_train_rows(4), out, store)
    assert _ids(out) == ["r-0", "r-2", "r-3"]
    failure = next(r for r in data.read_jsonl(store) if r["id"] == "r-1")
    assert failure["failed"] is True
    assert "chain" not in failure
    assert first["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]

    seen = _fake_generate_chain(monkeypatch)
    second = teacher.build_training_data(_train_rows(4), out, store)
    assert seen == []
    assert second["generated"] == 0
    assert second["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]
    assert _ids(out) == ["r-0", "r-2", "r-3"]


@pytest.mark.parametrize("field", ["question", "answer"])
def test_a_changed_row_for_a_stored_id_stops_before_any_call(
    tmp_path, monkeypatch, field
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    before = store.read_text(encoding="utf-8")
    changed = _train_rows(3)
    changed[0] = {**changed[0], field: "Something else entirely?"}
    seen = _fake_generate_chain(monkeypatch)
    with pytest.raises(ValueError, match="r-0"):
        teacher.build_training_data(changed, out, store)
    assert seen == []
    assert store.read_text(encoding="utf-8") == before


def test_rows_in_the_store_that_are_not_requested_are_kept_but_not_written(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    assert seen == []
    assert _ids(out) == ["r-0", "r-1"]
    assert _ids(store) == ["r-0", "r-1", "r-2", "r-3"]


def test_the_train_file_is_rewritten_freely_because_it_costs_nothing(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    assert _ids(out) == ["r-0", "r-1"]


def test_written_rows_keep_every_schema_field_and_a_chain(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    for row in data.read_jsonl(out):
        data.check_schema(row)
        assert row["chain"].startswith("CHAIN-")


def test_build_training_data_empty_input_raises_and_writes_nothing(
    tmp_path, monkeypatch
):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        teacher.build_training_data([], out, store)
    assert not out.exists()
    assert not store.exists()


def test_build_training_data_duplicate_ids_raise(tmp_path, monkeypatch):
    _fake_generate_chain(monkeypatch)
    out, store = _paths(tmp_path)
    rows = _train_rows(2)
    rows[1] = {**rows[1], "id": rows[0]["id"]}
    with pytest.raises(ValueError, match="id"):
        teacher.build_training_data(rows, out, store)


def test_build_training_data_progress_is_called_once_per_input_row(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    ticks = []
    teacher.build_training_data(
        _train_rows(4), out, store, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 4
    teacher.build_training_data(
        _train_rows(4), out, store, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 8


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


# ---- M5: make_variants (code-side filters on the model's rewordings) ----
LONG_Q = "If it takes 1 hour for 60 people to play an opera, how many hours will it take 600 people?"


def _source_row(**overrides):
    return _trow(
        1,
        question=LONG_Q,
        answer="1",
        accepted_alternatives=["1 hour"],
        chain=_teacher_chain("1"),
        **overrides,
    )


def _reworded(word):
    return LONG_Q.replace("play", word)


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


def test_variants_do_not_share_the_alternatives_list_with_the_source():
    row = _source_row()
    (variant,) = data.make_variants(row, [_reworded("perform")], cap=3)
    variant["accepted_alternatives"].append("changed")
    assert row["accepted_alternatives"] == ["1 hour"]


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


# ---- M5: paraphrase_question (the teacher writes the rewordings) ----
def _paraphrase_with(monkeypatch, *outputs):
    prompts, settings = [], []
    queue = list(outputs)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        settings.append((model, temperature))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, settings


def test_paraphrase_question_returns_the_rewordings_from_json(monkeypatch):
    prompts, settings = _paraphrase_with(
        monkeypatch, '{"rewordings": ["First way to ask?", "Second way to ask?"]}'
    )
    out = teacher.paraphrase_question("How do planes stay in the air?", 2)
    assert out == ["First way to ask?", "Second way to ask?"]
    assert "How do planes stay in the air?" in prompts[0]
    assert "2" in prompts[0]
    assert settings == [(config.TEACHER_MODEL, config.PARAPHRASE_TEMPERATURE)]


def test_paraphrase_prompt_asks_to_keep_numbers_units_and_negations(monkeypatch):
    prompts, _ = _paraphrase_with(monkeypatch, '{"rewordings": []}')
    teacher.paraphrase_question("Q?", 3)
    text = prompts[0].lower()
    assert "number" in text and "unit" in text and "negation" in text


def test_paraphrase_question_accepts_json_in_code_fences(monkeypatch):
    _paraphrase_with(monkeypatch, '```json\n{"rewordings": ["One way to ask?"]}\n```')
    assert teacher.paraphrase_question("Q?", 1) == ["One way to ask?"]


def test_paraphrase_question_keeps_only_non_blank_strings(monkeypatch):
    _paraphrase_with(monkeypatch, '{"rewordings": ["Good one?", 5, "", "  ", null]}')
    assert teacher.paraphrase_question("Q?", 3) == ["Good one?"]


@pytest.mark.parametrize(
    "reply",
    [
        "not json",
        "",
        "[]",
        '{"other": ["x"]}',
        '{"rewordings": "one string"}',
        '{"rewordings": null}',
    ],
)
def test_paraphrase_question_invalid_output_gives_no_candidates(monkeypatch, reply):
    _paraphrase_with(monkeypatch, reply)
    assert teacher.paraphrase_question("Q?", 3) == []


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated", "timeout", "server"])
def test_paraphrase_question_one_failed_call_gives_no_candidates(monkeypatch, kind):
    _paraphrase_with(monkeypatch, teacher.TeacherAPIError(kind))
    assert teacher.paraphrase_question("Q?", 3) == []


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit", "bad_request"])
def test_paraphrase_question_errors_that_would_repeat_stop_the_run(monkeypatch, kind):
    _paraphrase_with(monkeypatch, teacher.TeacherAPIError(kind, "why"))
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.paraphrase_question("Q?", 3)


# ---- M5: judge_variant ----
VARIANT_Q = "If it takes 1 hour for 60 people to perform an opera, how many hours will it take 600 people?"


def _judge(monkeypatch, *outputs):
    prompts, models = [], []
    queue = list(outputs)

    def fake(prompt, model, temperature):
        prompts.append(prompt)
        models.append((model, temperature))
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(teacher, "_generate", fake)
    return prompts, models


def test_judge_variant_keeps_a_natural_variant_with_the_same_answer(monkeypatch):
    prompts, models = _judge(monkeypatch, '{"natural": true, "answer_unchanged": true}')
    verdict = teacher.judge_variant(_source_row(), VARIANT_Q)
    assert verdict == {"keep": True, "reason": "kept"}
    assert LONG_Q in prompts[0] and VARIANT_Q in prompts[0]
    assert models == [(config.JUDGE_MODEL, config.JUDGE_TEMPERATURE)]


def test_judge_variant_prompt_shows_the_answer_but_not_the_chain(monkeypatch):
    prompts, _ = _judge(monkeypatch, '{"natural": true, "answer_unchanged": true}')
    row = _source_row()
    teacher.judge_variant(row, VARIANT_Q)
    assert row["answer"] in prompts[0]
    assert row["chain"] not in prompts[0]


@pytest.mark.parametrize(
    ("reply", "reason"),
    [
        ('{"natural": false, "answer_unchanged": true}', "not_natural"),
        ('{"natural": true, "answer_unchanged": false}', "answer_changed"),
        ('{"natural": false, "answer_unchanged": false}', "answer_changed"),
    ],
)
def test_judge_variant_drops_variants_the_judge_rejects(monkeypatch, reply, reason):
    _judge(monkeypatch, reply)
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": reason,
    }


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "",
        "[]",
        '{"natural": true}',
        '{"answer_unchanged": true}',
        '{"natural": "yes", "answer_unchanged": true}',
        '{"natural": true, "answer_unchanged": 1}',
        '{"natural": null, "answer_unchanged": null}',
    ],
)
def test_judge_variant_invalid_or_non_boolean_output_is_dropped(monkeypatch, reply):
    _judge(monkeypatch, reply)
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": "invalid_judge_output",
    }


def test_judge_variant_accepts_json_wrapped_in_code_fences(monkeypatch):
    _judge(monkeypatch, '```json\n{"natural": true, "answer_unchanged": true}\n```')
    assert teacher.judge_variant(_source_row(), VARIANT_Q)["keep"] is True


@pytest.mark.parametrize("kind", ["empty", "blocked", "truncated", "timeout", "server"])
def test_judge_variant_one_failed_call_drops_only_that_variant(monkeypatch, kind):
    _judge(monkeypatch, teacher.TeacherAPIError(kind))
    assert teacher.judge_variant(_source_row(), VARIANT_Q) == {
        "keep": False,
        "reason": "judge_error",
    }


@pytest.mark.parametrize("kind", ["auth", "quota", "rate_limit", "bad_request"])
def test_judge_variant_errors_that_would_repeat_stop_the_run(monkeypatch, kind):
    _judge(monkeypatch, teacher.TeacherAPIError(kind, "why"))
    with pytest.raises(teacher.TeacherAPIError, match="why"):
        teacher.judge_variant(_source_row(), VARIANT_Q)


# ---- M5: build_eval_data (permanent store; the paraphraser and judge are faked) ----
QUESTIONS = [
    "Why does the sky look blue during the day?",
    "How do planes stay in the air?",
    "What makes bread rise in the oven?",
]


def _qrow(i, **overrides):
    return _trow(i, question=QUESTIONS[i - 1], **overrides)


def _cands(question, n, start=0):
    return [f"{question[:-1]} indeed{'x' * (start + k)}?" for k in range(n)]


class _Spend:
    def __init__(self, verdicts=None, errors=()):
        self.verdicts = verdicts or {}
        self.errors = set(errors)
        self.asked = []
        self.judged = []
        self.calls_to_paraphrase = 0

    def paraphrase(self, question, n):
        self.asked.append((question, n))
        return _cands(question, n)

    def judge(self, source_row, variant_question):
        self.judged.append(variant_question)
        if variant_question in self.errors:
            return {"keep": False, "reason": "judge_error"}
        return self.verdicts.get(variant_question, {"keep": True, "reason": "kept"})


def _eval_paths(tmp_path):
    return (
        tmp_path / "eval.jsonl",
        tmp_path / "eval_review.jsonl",
        tmp_path / "eval_store.jsonl",
    )


def _build(tmp_path, monkeypatch, rows, spend=None, per_question=2, paraphrase=None):
    spend = spend or _Spend()
    monkeypatch.setattr(teacher, "judge_variant", spend.judge)
    out, review, store = _eval_paths(tmp_path)
    summary = teacher.build_eval_data(
        rows,
        out,
        review,
        store,
        per_question=per_question,
        paraphrase=paraphrase or spend.paraphrase,
    )
    return summary, spend


def test_build_eval_data_writes_kept_variants_with_their_verdict(tmp_path, monkeypatch):
    rows = [_qrow(1), _qrow(2)]
    summary, _ = _build(tmp_path, monkeypatch, rows)
    out, review, _ = _eval_paths(tmp_path)
    written = data.read_jsonl(out)
    assert [r["id"] for r in written] == ["t-1-v0", "t-1-v1", "t-2-v0", "t-2-v1"]
    assert all(r["verdict"] == {"keep": True, "reason": "kept"} for r in written)
    assert all(r["chain"] == rows[0]["chain"] for r in written[:2])
    assert (summary["kept"], summary["rejected"]) == (4, 0)
    assert data.read_jsonl(review) == []


def test_a_second_run_spends_nothing_and_gives_the_same_files(tmp_path, monkeypatch):
    rows = [_qrow(1), _qrow(2)]
    _build(tmp_path, monkeypatch, rows)
    out, review, _ = _eval_paths(tmp_path)
    first = (out.read_text(encoding="utf-8"), review.read_text(encoding="utf-8"))
    summary, spend = _build(tmp_path, monkeypatch, rows)
    assert spend.asked == []
    assert spend.judged == []
    assert (summary["paraphrase_calls"], summary["judge_calls"]) == (0, 0)
    assert (
        out.read_text(encoding="utf-8"),
        review.read_text(encoding="utf-8"),
    ) == first


def test_build_eval_data_asks_for_a_pool_larger_than_the_kept_target(
    tmp_path, monkeypatch
):
    _, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=2)
    assert spend.asked == [(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)]


def test_build_eval_data_stops_judging_a_question_once_it_has_enough(
    tmp_path, monkeypatch
):
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], per_question=2)
    assert len(spend.judged) == 2
    assert summary["judge_calls"] == 2
    assert summary["paraphrase_calls"] == 1


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


def test_build_eval_data_keeps_judging_past_rejections_until_enough(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    bad = {"keep": False, "reason": "not_natural"}
    spend = _Spend({pool[0]: bad, pool[1]: bad})
    _, spend = _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=2)
    out, review, _ = _eval_paths(tmp_path)
    assert len(spend.judged) == 4
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v2", "t-1-v3"]
    assert [r["id"] for r in data.read_jsonl(review)] == ["t-1-v0", "t-1-v1"]


def test_rejected_candidates_are_never_judged_again(tmp_path, monkeypatch):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    verdicts = {q: reject for q in pool}
    _build(tmp_path, monkeypatch, [_qrow(1)], _Spend(verdicts), per_question=2)
    spend = _Spend(verdicts)
    spend.paraphrase = lambda question, n: _cands(question, n, start=100)
    _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=2)
    assert not (set(spend.judged) & set(pool))


def test_an_exhausted_cache_asks_once_but_never_rejudges_known_candidates(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[0], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    _build(tmp_path, monkeypatch, [_qrow(1)], _Spend({q: reject for q in pool}))
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)])
    assert len(spend.asked) == 1
    assert spend.judged == []
    assert summary["paraphrase_calls"] == 1


def test_a_paraphraser_that_repeats_itself_cannot_loop(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, [_qrow(1)], per_question=1)
    spend = _Spend()
    spend.paraphrase = lambda question, n: _cands(question, n)
    summary, spend = _build(tmp_path, monkeypatch, [_qrow(1)], spend, per_question=9)
    assert summary["paraphrase_calls"] == 1


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


def test_a_changed_source_question_stops_before_any_call(tmp_path, monkeypatch):
    _build(tmp_path, monkeypatch, [_qrow(1)])
    spend = _Spend()
    with pytest.raises(ValueError, match="t-1"):
        _build(
            tmp_path,
            monkeypatch,
            [_trow(1, question="A completely different question text?")],
            spend,
        )
    assert spend.asked == [] and spend.judged == []


def test_build_eval_data_warns_about_a_question_with_no_kept_variants(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[1], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    spend = _Spend({q: reject for q in pool})
    with pytest.warns(UserWarning, match="t-2"):
        summary, _ = _build(tmp_path, monkeypatch, [_qrow(1), _qrow(2)], spend)
    assert summary["questions_without_variants"] == ["t-2"]


def test_build_eval_data_warns_about_a_category_with_no_kept_variants(
    tmp_path, monkeypatch
):
    pool = _cands(QUESTIONS[1], 2 * config.EVAL_OVERSAMPLE)
    reject = {"keep": False, "reason": "not_natural"}
    spend = _Spend({q: reject for q in pool})
    rows = [_qrow(1), _qrow(2, category="false_premise")]
    with pytest.warns(UserWarning, match="false_premise"):
        summary, _ = _build(tmp_path, monkeypatch, rows, spend)
    assert summary["categories_without_variants"] == ["false_premise"]


def test_build_eval_data_no_warning_when_every_question_has_a_variant(
    tmp_path, monkeypatch, recwarn
):
    _build(tmp_path, monkeypatch, [_qrow(1)])
    assert len(recwarn) == 0


def test_a_paraphraser_that_returns_nothing_warns(tmp_path, monkeypatch):
    with pytest.warns(UserWarning, match="t-1"):
        summary, spend = _build(
            tmp_path, monkeypatch, [_qrow(1)], paraphrase=lambda question, n: []
        )
    assert spend.judged == []
    assert summary["questions_without_variants"] == ["t-1"]


def test_build_eval_data_skips_short_questions_without_calling_the_paraphraser(
    tmp_path, monkeypatch
):
    short = _trow(1, question="Was Talos alive?")
    with pytest.warns(UserWarning, match="t-1"):
        _, spend = _build(tmp_path, monkeypatch, [short])
    assert spend.asked == []


def test_the_eval_files_are_rewritten_freely_because_it_costs_nothing(
    tmp_path, monkeypatch
):
    out, _, _ = _eval_paths(tmp_path)
    out.write_text("old\n", encoding="utf-8")
    _build(tmp_path, monkeypatch, [_qrow(1)])
    assert [r["id"] for r in data.read_jsonl(out)] == ["t-1-v0", "t-1-v1"]


@pytest.mark.parametrize("stage", ["judge", "paraphrase"])
def test_a_stopping_error_keeps_the_store_progress_and_writes_no_outputs(
    tmp_path, monkeypatch, stage
):
    def boom(*args):
        raise teacher.TeacherAPIError("quota", "daily limit")

    spend = _Spend()
    if stage == "judge":
        calls = {"n": 0}

        def judge_then_boom(source_row, variant_question):
            calls["n"] += 1
            if calls["n"] == 2:
                boom()
            return {"keep": True, "reason": "kept"}

        monkeypatch.setattr(teacher, "judge_variant", judge_then_boom)
        paraphrase = spend.paraphrase
    else:
        monkeypatch.setattr(teacher, "judge_variant", spend.judge)
        paraphrase = boom
    out, review, store = _eval_paths(tmp_path)
    with pytest.raises(teacher.TeacherAPIError, match="daily limit"):
        teacher.build_eval_data(
            [_qrow(1)], out, review, store, per_question=2, paraphrase=paraphrase
        )
    assert not out.exists()
    assert not review.exists()
    if stage == "judge":
        record = data.read_jsonl(store)[0]
        assert record["candidates"][0]["verdict"] == {"keep": True, "reason": "kept"}


def test_build_eval_data_progress_ticks_for_every_candidate_it_goes_through(
    tmp_path, monkeypatch
):
    ticks = []
    spend = _Spend()
    monkeypatch.setattr(teacher, "judge_variant", spend.judge)
    out, review, store = _eval_paths(tmp_path)
    for _ in range(2):
        teacher.build_eval_data(
            [_qrow(1)], out, review, store, per_question=2,
            progress=lambda: ticks.append(1), paraphrase=spend.paraphrase,
        )  # fmt: skip
    assert len(ticks) == 4


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


# ---- M5: eval-data command ----


def test_pipeline_eval_data_invalid_test_set_fails_with_the_row_id(tmp_path):
    test_path = tmp_path / "test_set.jsonl"
    data.write_jsonl(test_path, [_trow(1, chain=_teacher_chain("12"))])
    result = _run_pipeline(
        "eval-data", "--test-set", str(test_path), "--out", str(tmp_path / "e.jsonl")
    )
    assert result.returncode != 0
    assert "t-1" in result.stderr


# ---- rate limits: wait for the delay the server asks for ----
def _rate_limit_error(delay):
    details = [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}
    ]
    return _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED", details)


@pytest.mark.parametrize(
    ("delay", "seconds"), [("23s", 23.0), ("1.5s", 1.5), ("81429s", 81429.0)]
)
def test_call_model_reads_the_retry_delay_the_server_asks_for(
    monkeypatch, delay, seconds
):
    _use_client(monkeypatch, _rate_limit_error(delay))
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.kind == "rate_limit"
    assert caught.value.retry_after == seconds


def test_call_model_without_retry_info_has_no_retry_after(monkeypatch):
    _use_client(monkeypatch, _api_error(errors.ClientError, 429, "RESOURCE_EXHAUSTED"))
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._call_model("m", "p", 0.0)
    assert caught.value.retry_after is None


def test_generate_waits_the_servers_delay_when_it_is_longer_than_the_default(
    monkeypatch,
):
    _, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError("rate_limit", retry_after=23.0), "fine"]
    )
    assert teacher._generate("p", "m", 0.0) == "fine"
    assert sleeps == [23.0]


def test_generate_uses_the_default_wait_when_the_servers_delay_is_shorter(monkeypatch):
    _, sleeps = _script_call_model(
        monkeypatch, [teacher.TeacherAPIError("rate_limit", retry_after=0.1), "fine"]
    )
    teacher._generate("p", "m", 0.0)
    assert sleeps == [config.TEACHER_RETRY_SECONDS]


def test_generate_does_not_wait_when_the_delay_is_longer_than_the_maximum(monkeypatch):
    too_long = config.TEACHER_MAX_WAIT_SECONDS + 1
    calls, sleeps = _script_call_model(
        monkeypatch,
        [teacher.TeacherAPIError("rate_limit", retry_after=too_long), "never"],
    )
    with pytest.raises(teacher.TeacherAPIError) as caught:
        teacher._generate("p", "m", 0.0)
    assert caught.value.retry_after == too_long
    assert len(calls) == 1
    assert sleeps == []


def test_generate_waits_exactly_the_maximum_delay_allowed(monkeypatch):
    _, sleeps = _script_call_model(
        monkeypatch,
        [
            teacher.TeacherAPIError(
                "rate_limit", retry_after=config.TEACHER_MAX_WAIT_SECONDS
            ),
            "fine",
        ],
    )
    teacher._generate("p", "m", 0.0)
    assert sleeps == [config.TEACHER_MAX_WAIT_SECONDS]


# ---- timing of build_training_data (only rows that were generated count) ----
def test_build_training_data_summary_has_row_seconds(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert summary["row_seconds"] == {"n": 4, "mean": 1.0, "max": 1.0}


def test_build_training_data_row_seconds_count_only_generated_rows(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    _fake_generate_chain(monkeypatch)
    _fake_clock(monkeypatch, 1.0)
    summary = teacher.build_training_data(_train_rows(4), out, store)
    assert summary["row_seconds"] == {"n": 2, "mean": 1.0, "max": 1.0}


def test_build_training_data_row_seconds_when_nothing_was_generated(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["row_seconds"] == {"n": 0, "mean": None, "max": None}


def test_build_training_data_timing_covers_only_this_run(tmp_path, monkeypatch):
    _script_call_model(monkeypatch, ["old call"])
    teacher._generate("p", "m", 0.0)
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["timing"] == {}


def test_build_training_data_timing_reports_the_api_calls_it_made(
    tmp_path, monkeypatch
):
    _script_call_model(monkeypatch, ["fine", "fine"])

    def fake_generate_chain(row):
        teacher._generate("p", "m-1", 0.0)
        return {**row, "chain": "c"}

    monkeypatch.setattr(teacher, "generate_chain", fake_generate_chain)
    out, store = _paths(tmp_path)
    summary = teacher.build_training_data(_train_rows(2), out, store)
    assert summary["timing"]["m-1"]["calls"] == 2
    assert summary["timing"]["m-1"]["outcomes"] == {"ok": 2}


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


# ---- M6: token audit ----
def _tiny_tokenizer(with_bos=True):
    tok = Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(["<unk>", "<bos>", config.EOS_TOKEN])
    if with_bos:
        tok.post_processor = processors.TemplateProcessing(
            single="<bos> $A", special_tokens=[("<bos>", tok.token_to_id("<bos>"))]
        )
    return tok


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


# ---- M6: train.py (memory estimate and GPU choice) ----
def test_memory_estimate_has_positive_parts_that_sum_to_the_total():
    parts = train.estimate_training_memory_gib(256, batch_size=1)
    pieces = ("weights", "lora_and_optimizer", "activations", "logits", "overhead")
    assert all(parts[name] > 0 for name in pieces)
    assert parts["total"] == pytest.approx(sum(parts[name] for name in pieces))


def test_memory_estimate_grows_with_tokens_and_batch_size():
    small = train.estimate_training_memory_gib(256, batch_size=1)["total"]
    longer = train.estimate_training_memory_gib(1024, batch_size=1)["total"]
    wider = train.estimate_training_memory_gib(256, batch_size=4)["total"]
    assert small < longer
    assert small < wider


def test_memory_estimate_logits_scale_linearly_with_tokens():
    short = train.estimate_training_memory_gib(256)["logits"]
    long = train.estimate_training_memory_gib(512)["logits"]
    assert long == pytest.approx(2 * short)


def test_memory_estimate_for_short_examples_is_a_few_gib():
    assert 1 < train.estimate_training_memory_gib(256)["total"] < 8


@pytest.mark.parametrize("tokens, batch", [(0, 1), (-1, 1), (256, 0), (256, -2)])
def test_memory_estimate_rejects_non_positive_inputs(tokens, batch):
    with pytest.raises(ValueError):
        train.estimate_training_memory_gib(tokens, batch_size=batch)


OPTIONS = {
    "A": {"vram_gib": 16, "usd_per_second": 0.3},
    "B": {"vram_gib": 24, "usd_per_second": 0.2},
    "C": {"vram_gib": 48, "usd_per_second": 0.5},
}


def test_choose_gpu_picks_the_cheapest_that_fits_not_the_smallest():
    assert train.choose_gpu(10, OPTIONS) == "B"


def test_choose_gpu_only_trusts_a_fraction_of_the_memory():
    assert train.choose_gpu(20, OPTIONS) == "B"  # 24 * 0.85 = 20.4
    assert train.choose_gpu(21, OPTIONS) == "C"


def test_choose_gpu_with_nothing_big_enough_raises():
    with pytest.raises(ValueError, match="fits"):
        train.choose_gpu(100, OPTIONS)


def test_choose_gpu_breaks_a_price_tie_with_the_smaller_memory():
    tied = {
        "big": {"vram_gib": 48, "usd_per_second": 0.2},
        "small": {"vram_gib": 24, "usd_per_second": 0.2},
    }
    assert train.choose_gpu(5, tied) == "small"


@pytest.mark.parametrize(
    ("needed", "expected"), [(5, "T4"), (20, "L4"), (30, "L40S"), (60, "A100-80GB")]
)
def test_choose_gpu_with_the_real_modal_prices(needed, expected):
    assert train.choose_gpu(needed) == expected


def test_every_gpu_option_has_a_positive_size_and_price():
    for name, option in config.GPU_OPTIONS.items():
        assert option["vram_gib"] > 0, name
        assert option["usd_per_second"] > 0, name


# ---- M6: validate --tokens ----
def _tiny_tokenizer_file(tmp_path):
    path = tmp_path / "tok.json"
    _tiny_tokenizer().save(str(path))
    return path


def _validate_tokens(tmp_path, *extra, eval_rows=()):
    paths = _write_all(
        tmp_path, [_trow(1)], [_train_row(1), _train_row(2)], list(eval_rows)
    )
    test_path, train_path, eval_path, hash_path = paths
    return _run_pipeline(
        "validate",
        "--test-set", str(test_path),
        "--train", str(train_path),
        "--eval", str(eval_path),
        "--hash", str(hash_path),
        "--tokens",
        "--tokenizer", str(_tiny_tokenizer_file(tmp_path)),
        "--token-audit", str(tmp_path / "audit.json"),
        *extra,
    )  # fmt: skip


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


def test_the_chosen_gpu_is_one_of_the_known_options():
    assert config.GPU_TYPE in config.GPU_OPTIONS


def test_pipeline_validate_tokens_reports_the_chosen_gpu_and_that_it_fits(tmp_path):
    result = _validate_tokens(tmp_path, "--max-seq-length", "1000")
    assert result.returncode == 0, result.stderr
    assert f"chosen GPU: {config.GPU_TYPE}" in result.stdout
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert audit["gpu_type"] == config.GPU_TYPE
    assert audit["fits_chosen_gpu"] is True


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


# ---- M7: cut_at_final_answer ----
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (_teacher_chain("9") + "\n\nQuestion: What next?\nRestate: more", _teacher_chain("9")),
        ("Restate: x\nFINAL ANSWER: 9\nand then rambling", "Restate: x\nFINAL ANSWER: 9"),
        ("Restate: x\nFinal   answer : 9\n### Instruction:\nfoo", "Restate: x\nFinal   answer : 9"),
        ("Final answer: 9", "Final answer: 9"),
        ("Final answer: 9\nFinal answer: 10", "Final answer: 9"),
        ("Final answer: 9   \nmore", "Final answer: 9"),
        ("Restate: the Final answer: is tricky\nSolve: s\nFinal answer: 9\nextra", "Restate: the Final answer: is tricky\nSolve: s\nFinal answer: 9"),
        ("Restate: no label here   \n", "Restate: no label here"),
        ("", ""),
    ],
)  # fmt: skip
def test_cut_at_final_answer(text, expected):
    assert evaluation.cut_at_final_answer(text) == expected


# ---- M7: make_generate_fn (a numpy fake stands in for the model and tokenizer) ----
class _FakeTokenizer:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self.decoded = []
        self.sides = []

    def __call__(self, prompts, return_tensors=None, padding=True):
        self.sides.append(self.padding_side)
        ids = [[ord(c) % 200 + 1 for c in prompt] for prompt in prompts]
        width = max(len(row) for row in ids)
        pad = lambda row: (
            ([0] * (width - len(row)) + row)
            if self.padding_side == "left"
            else (row + [0] * (width - len(row)))
        )
        input_ids = np.array([pad(row) for row in ids])
        return {"input_ids": input_ids, "attention_mask": (input_ids != 0).astype(int)}

    def batch_decode(self, token_ids, skip_special_tokens=True):
        self.decoded.append(np.array(token_ids))
        return [self.scripted.pop(0) for _ in range(len(token_ids))]


class _FakeModel:
    device = "cpu"

    def __init__(self, new_tokens=4):
        self.new_tokens = new_tokens
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        new = np.full((input_ids.shape[0], self.new_tokens), 7)
        return np.concatenate([input_ids, new], axis=1)


def test_generate_fn_returns_one_output_per_prompt_in_order():
    tok = _FakeTokenizer(["A", "B", "C"])
    generate = evaluation.make_generate_fn(
        _FakeModel(), tok, max_new_tokens=5, batch_size=8
    )
    assert generate(["one", "three", "five!"]) == ["A", "B", "C"]


def test_generate_fn_splits_into_batches_and_keeps_the_order():
    tok = _FakeTokenizer(list("abcde"))
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=2)
    assert generate(["p1", "p2", "p3", "p4", "p5"]) == list("abcde")
    assert [call["input_ids"].shape[0] for call in model.calls] == [2, 2, 1]


def test_generate_fn_with_a_batch_larger_than_the_data_makes_one_call():
    tok = _FakeTokenizer(["x", "y"])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=64)
    generate(["a", "b"])
    assert len(model.calls) == 1


def test_generate_fn_with_no_prompts_makes_no_calls():
    tok = _FakeTokenizer([])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=2)
    assert generate([]) == []
    assert model.calls == []


def test_generate_fn_pads_on_the_left_so_different_lengths_share_a_batch():
    tok = _FakeTokenizer(["x", "y"])
    model = _FakeModel()
    generate = evaluation.make_generate_fn(model, tok, max_new_tokens=5, batch_size=8)
    generate(["short", "a much longer prompt"])
    assert tok.sides == ["left"]
    mask = model.calls[0]["attention_mask"]
    assert mask[0][0] == 0 and mask[0][-1] == 1  # the short prompt is padded first
    assert mask[1][0] == 1


def test_generate_fn_decodes_only_the_new_tokens_never_the_prompt():
    tok = _FakeTokenizer(["x"])
    generate = evaluation.make_generate_fn(
        _FakeModel(new_tokens=4), tok, max_new_tokens=5, batch_size=8
    )
    generate(["a prompt of some length"])
    assert tok.decoded[0].shape == (1, 4)
    assert (tok.decoded[0] == 7).all()


def test_generate_fn_is_greedy_and_passes_the_token_cap():
    tok = _FakeTokenizer(["x"])
    model = _FakeModel()
    evaluation.make_generate_fn(model, tok, max_new_tokens=123, batch_size=8)(["p"])
    call = model.calls[0]
    assert call["max_new_tokens"] == 123
    assert call["do_sample"] is False


def test_generate_fn_uses_the_configured_defaults():
    tok = _FakeTokenizer(["x"])
    model = _FakeModel()
    evaluation.make_generate_fn(model, tok)(["p"])
    assert model.calls[0]["max_new_tokens"] == config.EVAL_MAX_NEW_TOKENS


def test_generate_fn_gives_a_pad_token_when_the_tokenizer_has_none():
    tok = _FakeTokenizer(["x"])
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.pad_token == tok.eos_token


def test_generate_fn_cuts_each_output_at_its_final_answer_line():
    tok = _FakeTokenizer([_teacher_chain("9") + "\n\nQuestion: next one?"])
    generate = evaluation.make_generate_fn(
        _FakeModel(), tok, max_new_tokens=5, batch_size=8
    )
    assert generate(["p"]) == [_teacher_chain("9")]


def test_generate_fn_gives_identical_outputs_on_a_repeated_greedy_run():
    first = evaluation.make_generate_fn(
        _FakeModel(), _FakeTokenizer(["same", "same"]), max_new_tokens=5, batch_size=8
    )
    second = evaluation.make_generate_fn(
        _FakeModel(), _FakeTokenizer(["same", "same"]), max_new_tokens=5, batch_size=8
    )
    assert first(["p", "q"]) == second(["p", "q"])


# ---- M7: grade ----
def _grade_row(**overrides):
    return _trow(1, answer="9", accepted_alternatives=["nine"], **overrides)


def test_grade_a_correct_well_formed_answer():
    graded = evaluation.grade(_grade_row(), _teacher_chain("9"))
    assert (graded["correct"], graded["format_ok"], graded["final_answer"]) == (
        True,
        True,
        "9",
    )
    assert graded["id"] == "t-1"
    assert graded["category"] == "lateral_thinking"


def test_grade_a_wrong_but_well_formed_answer():
    graded = evaluation.grade(_grade_row(), _teacher_chain("12"))
    assert (graded["correct"], graded["format_ok"]) == (False, True)


def test_grade_accepts_an_alternative_answer():
    assert evaluation.grade(_grade_row(), _teacher_chain("nine"))["correct"] is True


def test_grade_output_cut_off_before_the_final_answer_is_wrong_and_a_format_failure():
    truncated = "\n".join(_teacher_chain("9").split("\n")[:3])
    graded = evaluation.grade(_grade_row(), truncated)
    assert graded["final_answer"] is None
    assert (graded["correct"], graded["format_ok"]) == (False, False)


def test_grade_right_answer_with_a_broken_format_counts_as_correct_but_not_compliant():
    broken = "Final answer: 9"
    graded = evaluation.grade(_grade_row(), broken)
    assert (graded["correct"], graded["format_ok"]) == (True, False)


@pytest.mark.parametrize("output", ["", "   ", "no label anywhere", "Final answer:"])
def test_grade_unparsable_output_is_wrong_and_not_compliant(output):
    graded = evaluation.grade(_grade_row(), output)
    assert (graded["correct"], graded["format_ok"]) == (False, False)


def test_grade_records_the_response_length():
    output = _teacher_chain("9")
    assert evaluation.grade(_grade_row(), output)["length"] == len(output)


def test_grade_keeps_the_source_id_of_an_eval_variant():
    variant = _grade_row(id="t-1-v0", source_id="t-1")
    assert evaluation.grade(variant, _teacher_chain("9"))["source_id"] == "t-1"


# ---- M7: score ----
def _rows_by_category(counts):
    rows = []
    for category, n in counts.items():
        for i in range(n):
            rows.append(_grade_row(id=f"{category}-{i}", category=category))
    return rows


def test_score_of_no_rows_is_undefined_everywhere_not_zero():
    metrics = evaluation.score([], [])
    assert metrics["n"] == 0
    assert metrics["accuracy"] == "undefined"
    assert metrics["format_compliance"] == "undefined"
    assert metrics["avg_response_chars"] == "undefined"
    assert all(v["accuracy"] == "undefined" for v in metrics["per_category"].values())


def test_score_a_category_with_no_rows_reports_undefined_not_zero():
    rows = _rows_by_category({"false_premise": 2})
    metrics = evaluation.score(rows, [_teacher_chain("9"), _teacher_chain("12")])
    assert metrics["per_category"]["false_premise"] == {
        "n": 2,
        "correct": 1,
        "accuracy": 0.5,
    }
    assert metrics["per_category"]["time_sensitive"]["accuracy"] == "undefined"
    assert metrics["per_category"]["time_sensitive"]["n"] == 0


def test_score_overall_numbers_and_category_order():
    rows = _rows_by_category({"false_premise": 2, "misconception": 2})
    outputs = [
        _teacher_chain("9"),
        _teacher_chain("9"),
        _teacher_chain("12"),
        "garbage",
    ]
    metrics = evaluation.score(rows, outputs)
    assert metrics["n"] == 4
    assert metrics["accuracy"] == 0.5
    assert metrics["format_compliance"] == 0.75
    assert metrics["avg_response_chars"] == pytest.approx(
        sum(len(o) for o in outputs) / 4
    )
    assert list(metrics["per_category"]) == config.CATEGORIES


def test_score_counts_unparsable_outputs_as_wrong():
    rows = _rows_by_category({"misconception": 2})
    metrics = evaluation.score(rows, ["", "nothing"])
    assert metrics["accuracy"] == 0.0
    assert metrics["per_category"]["misconception"]["correct"] == 0


def test_score_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="2 rows but 1 outputs"):
        evaluation.score(_rows_by_category({"misconception": 2}), [_teacher_chain("9")])


def test_score_a_category_outside_the_known_list_raises():
    rows = [_grade_row(category="character_counting")]
    with pytest.raises(ValueError, match="character_counting"):
        evaluation.score(rows, [_teacher_chain("9")])


# ---- M7: evaluate ----
def test_evaluate_builds_one_prompt_per_row_and_scores_the_outputs():
    rows = _rows_by_category({"false_premise": 1, "misconception": 1})
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return [_teacher_chain("9"), _teacher_chain("12")]

    results, metrics = evaluation.evaluate(
        rows, generate, lambda row: f"PROMPT {row['id']}"
    )
    assert seen == ["PROMPT false_premise-0", "PROMPT misconception-0"]
    assert [r["correct"] for r in results] == [True, False]
    assert metrics["accuracy"] == 0.5


def test_evaluate_with_no_rows_makes_no_generation_call():
    calls = []
    results, metrics = evaluation.evaluate(
        [], lambda prompts: calls.append(prompts) or [], lambda row: "p"
    )
    assert (results, calls) == ([], [])
    assert metrics["accuracy"] == "undefined"


def test_evaluate_a_generator_that_returns_the_wrong_number_of_outputs_raises():
    rows = _rows_by_category({"misconception": 2})
    with pytest.raises(ValueError, match="2 rows but 1 outputs"):
        evaluation.evaluate(rows, lambda prompts: ["only one"], lambda row: "p")


# ---- M7: few-shot examples and prompts (training data only) ----


def test_finetuned_prompt_is_the_plain_training_prompt():
    row = _grade_row(question="Just the question please?")
    assert evaluation.finetuned_prompt(row) == data.format_prompt(
        "Just the question please?"
    )


def test_check_prompt_length_accepts_prompts_at_the_limit():
    tok = _tiny_tokenizer()
    prompt = "a b c"
    limit = len(tok.encode(prompt, add_special_tokens=True).ids)
    assert evaluation.check_prompt_length([prompt], tok, limit) is None


def test_check_prompt_length_names_the_longest_prompt_over_the_limit():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="2 prompts") as caught:
        evaluation.check_prompt_length(
            ["a b", "a b c d e f g", "a b c d e f g h"], tok, 6
        )
    assert "longest" in str(caught.value)


# ---- M7: weak categories ----
def _acc(**values):
    return values


COUNTS = {c: 5 for c in config.CATEGORIES}


def test_weak_categories_all_equal_flags_nothing():
    accuracy = {c: 0.8 for c in config.CATEGORIES}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.1) == []


def test_weak_categories_one_eligible_category_has_nothing_to_compare_against():
    assert (
        evaluation.find_weak_categories(
            {"misconception": 0.1}, {"misconception": 5}, gap=0.1
        )
        == []
    )


def test_weak_categories_flags_a_category_that_lags_the_average():
    accuracy = {"false_premise": 0.9, "misconception": 0.9, "pun_ambiguity": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "pun_ambiguity"
    ]


def test_weak_categories_a_gap_exactly_at_the_threshold_is_flagged():
    accuracy = {"false_premise": 1.0, "misconception": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.25) == [
        "misconception"
    ]


def test_weak_categories_a_gap_just_under_the_threshold_is_not_flagged():
    accuracy = {"false_premise": 1.0, "misconception": 0.5}
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.26) == []


def test_weak_categories_ties_are_ordered_by_accuracy_then_name():
    accuracy = {
        "false_premise": 1.0,
        "misconception": 0.4,
        "pun_ambiguity": 0.4,
        "lateral_thinking": 0.2,
        "time_sensitive": 1.0,
    }
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "lateral_thinking",
        "misconception",
        "pun_ambiguity",
    ]


def test_weak_categories_below_the_minimum_source_count_are_not_flagged_or_averaged():
    accuracy = {"false_premise": 0.9, "misconception": 0.9, "pun_ambiguity": 0.0}
    counts = {"false_premise": 5, "misconception": 5, "pun_ambiguity": 2}
    assert evaluation.find_weak_categories(accuracy, counts, gap=0.1) == []


def test_weak_categories_ignore_nan_and_undefined_accuracies():
    accuracy = {
        "false_premise": 0.9,
        "misconception": float("nan"),
        "pun_ambiguity": "undefined",
        "lateral_thinking": 0.3,
    }
    assert evaluation.find_weak_categories(accuracy, COUNTS, gap=0.2) == [
        "lateral_thinking"
    ]


def test_weak_categories_a_category_missing_from_the_counts_is_not_eligible():
    accuracy = {"false_premise": 0.9, "misconception": 0.1}
    assert (
        evaluation.find_weak_categories(accuracy, {"false_premise": 5}, gap=0.1) == []
    )


@pytest.mark.parametrize("gap", [0, -0.1])
def test_weak_categories_needs_a_positive_gap(gap):
    with pytest.raises(ValueError, match="gap"):
        evaluation.find_weak_categories({"misconception": 0.5}, COUNTS, gap=gap)


# ---- M7: timer, versions, GPU summary, result record ----
def test_timer_records_the_elapsed_time(monkeypatch):
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(evaluation, "_now", lambda: next(ticks))
    with evaluation.Timer() as timer:
        pass
    assert timer.seconds == 2.5


def test_timer_records_the_time_even_if_the_block_raises(monkeypatch):
    ticks = iter([1.0, 4.0])
    monkeypatch.setattr(evaluation, "_now", lambda: next(ticks))
    timer = evaluation.Timer()
    with pytest.raises(RuntimeError, match="boom"), timer:
        raise RuntimeError("boom")
    assert timer.seconds == 3.0


def test_package_versions_reports_installed_missing_and_python():
    versions = evaluation.package_versions(["numpy", "definitely-not-a-package-xyz"])
    assert versions["numpy"] == np.__version__
    assert versions["definitely-not-a-package-xyz"] is None
    assert versions["python"].count(".") >= 1


def test_gpu_summary_has_none_fields_when_there_is_no_gpu():
    def no_smi():
        raise FileNotFoundError("nvidia-smi")

    with evaluation.GpuMonitor(run_smi=no_smi, torch_peak_mib=lambda: None) as monitor:
        pass
    assert monitor.summary() == {
        "gpu_name": None,
        "peak_memory_mib": None,
        "mean_utilization_percent": None,
        "peak_torch_memory_allocated_mib": None,
    }


def test_gpu_summary_takes_the_peak_memory_and_mean_utilization_over_the_samples():
    readings = iter(
        ["NVIDIA L4, 1000, 40\n", "NVIDIA L4, 2000, 80\n", "NVIDIA L4, 1500, 60\n"]
    )
    monitor = evaluation.GpuMonitor(
        run_smi=lambda: next(readings), torch_peak_mib=lambda: 1234.5
    )
    for _ in range(3):
        monitor.sample_once()
    assert monitor.summary() == {
        "gpu_name": "NVIDIA L4",
        "peak_memory_mib": 2000,
        "mean_utilization_percent": 60.0,
        "peak_torch_memory_allocated_mib": 1234.5,
    }


def test_gpu_monitor_ignores_unreadable_smi_output():
    outputs = iter(["garbage", "", "NVIDIA L4, 500, 10"])
    monitor = evaluation.GpuMonitor(
        run_smi=lambda: next(outputs), torch_peak_mib=lambda: None
    )
    for _ in range(3):
        monitor.sample_once()
    assert monitor.summary()["peak_memory_mib"] == 500


def test_gpu_monitor_samples_on_entry_and_exit():
    count = []

    def smi():
        count.append(1)
        return "NVIDIA L4, 100, 5"

    with evaluation.GpuMonitor(run_smi=smi, torch_peak_mib=lambda: None, interval=10):
        pass
    assert len(count) >= 2


def _result(**overrides):
    base = {
        "split": "eval",
        "metrics": evaluation.score([], []),
        "rows": [_grade_row()],
        "batch_size": 8,
        "max_seq_length": 2048,
        "versions": {"python": "3.14"},
        "eval_seconds": 12.0,
        "gpu_summary": {"gpu_name": None},
        "prompt_style": "zero_shot",
        "max_new_tokens": 384,
    }
    return evaluation.build_result(**{**base, **overrides})


def test_build_result_has_every_required_field_and_is_json_serializable():
    result = _result()
    assert all(field in result for field in evaluation.REQUIRED_RESULT_FIELDS)
    assert result["model_name"] == config.MODEL_NAME
    assert result["test_size"] == 1
    assert json.loads(json.dumps(result)) == result


def test_build_result_records_the_split_prompt_style_and_generation_settings():
    result = _result()
    assert (result["split"], result["prompt_style"]) == ("eval", "zero_shot")
    assert result["max_new_tokens"] == 384
    assert result["gpu_type"] == config.GPU_TYPE
    assert "fewshot_ids" not in result


def test_save_result_writes_json_atomically(tmp_path):
    path = tmp_path / "results" / "baseline.json"
    evaluation.save_result(path, _result())
    assert json.loads(path.read_text(encoding="utf-8"))["split"] == "eval"
    assert list(path.parent.iterdir()) == [path]


# ---- M7: the final test is run once ----
def test_the_test_split_is_refused_a_second_time_without_force(tmp_path):
    path = tmp_path / "final_test.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        evaluation.ensure_split_allowed("test", path, force=False)


def test_the_test_split_is_allowed_the_first_time_and_with_force(tmp_path):
    path = tmp_path / "final_test.json"
    assert evaluation.ensure_split_allowed("test", path, force=False) is None
    path.write_text("{}", encoding="utf-8")
    assert evaluation.ensure_split_allowed("test", path, force=True) is None


def test_the_eval_split_may_be_rerun(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{}", encoding="utf-8")
    assert evaluation.ensure_split_allowed("eval", path, force=False) is None


def test_an_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        evaluation.ensure_split_allowed("train", tmp_path / "x.json", force=False)


# ---- M7: the baseline prompt is zero-shot (a written instruction, no examples) ----
def test_baseline_prompt_has_the_instruction_the_question_and_the_response_marker():
    prompt = evaluation.baseline_prompt(
        _grade_row(question="A brand new question here?")
    )
    assert prompt.startswith(config.INSTRUCTION_MARKER)
    assert prompt.endswith(config.RESPONSE_MARKER)
    assert "A brand new question here?" in prompt
    assert all(label in prompt for label in config.SECTION_LABELS)


def test_baseline_prompt_contains_no_examples():
    prompt = evaluation.baseline_prompt(
        _grade_row(question="A brand new question here?")
    )
    assert prompt.count(config.RESPONSE_MARKER) == 1
    assert prompt.count(config.INSTRUCTION_MARKER) == 1


def test_baseline_prompt_is_the_same_for_the_same_question_and_differs_from_the_trained_prompt():
    row = _grade_row(question="Same question each time?")
    assert evaluation.baseline_prompt(row) == evaluation.baseline_prompt(dict(row))
    assert evaluation.baseline_prompt(row) != evaluation.finetuned_prompt(row)


def test_baseline_prompt_refuses_a_question_containing_a_template_marker():
    with pytest.raises(ValueError):
        evaluation.baseline_prompt(
            _grade_row(question=f"What is {config.RESPONSE_MARKER} this?")
        )


def test_the_teacher_and_the_baseline_share_one_instruction_text():
    assert config.CHAIN_INSTRUCTIONS in teacher.chain_prompt("Q?", "a")
    assert config.CHAIN_INSTRUCTIONS in evaluation.baseline_prompt(_grade_row())


# ---- M7: where a run writes and which rows each split reads ----
def test_default_output_path_for_the_baseline_on_the_eval_split():
    assert evaluation.default_output_path("eval", "baseline") == config.BASELINE_PATH


def test_default_output_path_for_the_final_test_with_the_trained_model():
    assert evaluation.default_output_path("test", "finetuned") == config.FINAL_TEST_PATH


def test_an_explicit_output_always_wins(tmp_path):
    chosen = tmp_path / "mine.json"
    assert evaluation.default_output_path("test", "baseline", str(chosen)) == chosen


@pytest.mark.parametrize(
    ("split", "style"), [("eval", "finetuned"), ("test", "baseline")]
)
def test_other_combinations_must_name_their_output(split, style):
    with pytest.raises(ValueError, match="--output"):
        evaluation.default_output_path(split, style)


def test_load_split_rows_reads_the_eval_file(monkeypatch, tmp_path):
    path = tmp_path / "eval.jsonl"
    data.write_jsonl(path, [_trow(1), _trow(2)])
    monkeypatch.setattr(config, "EVAL_PATH", path)
    assert [r["id"] for r in evaluation.load_split_rows("eval")] == ["t-1", "t-2"]


def test_load_split_rows_reads_the_validated_test_set(monkeypatch, tmp_path):
    path = _write_test_set(tmp_path, [_trow(1)])
    monkeypatch.setattr(config, "TEST_SET_PATH", path)
    monkeypatch.setattr(config, "EXCLUDED_TEST_IDS", frozenset())
    assert [r["id"] for r in evaluation.load_split_rows("test")] == ["t-1"]


def test_load_split_rows_refuses_an_invalid_test_set(monkeypatch, tmp_path):
    path = _write_test_set(tmp_path, [_trow(1, chain=_teacher_chain("12"))])
    monkeypatch.setattr(config, "TEST_SET_PATH", path)
    with pytest.raises(ValueError, match="t-1"):
        evaluation.load_split_rows("test")


def test_load_split_rows_with_an_empty_eval_file_raises(monkeypatch, tmp_path):
    path = tmp_path / "eval.jsonl"
    data.write_jsonl(path, [])
    monkeypatch.setattr(config, "EVAL_PATH", path)
    with pytest.raises(ValueError, match="no rows"):
        evaluation.load_split_rows("eval")


def test_load_split_rows_rejects_an_unknown_split():
    with pytest.raises(ValueError, match="split"):
        evaluation.load_split_rows("train")


# ---- the Modal container runs Python 3.11; the files it imports must parse there ----
_UNPARENTHESIZED_EXCEPT = r"^\s*except\s+[^(:\n]+,[^:\n]+:\s*$"


@pytest.mark.parametrize(
    "name",
    [
        "src/config.py",
        "src/data.py",
        "src/eval.py",
        "src/train.py",
        "scripts/modal_app.py",
    ],
)
def test_container_imported_files_avoid_syntax_newer_than_python_3_11(name):
    import re

    source = (ROOT / name).read_text(encoding="utf-8")
    offenders = [
        line for line in source.splitlines() if re.match(_UNPARENTHESIZED_EXCEPT, line)
    ]
    assert offenders == []


# ---- build_training_data without generating: use only the chains already stored ----
def test_no_generate_writes_only_stored_rows_and_makes_no_calls(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(4), out, store)
    seen = _fake_generate_chain(monkeypatch)
    summary = teacher.build_training_data(_train_rows(6), out, store, generate=False)
    assert seen == []
    assert _ids(out) == ["r-0", "r-1", "r-2", "r-3"]
    assert (summary["written"], summary["pending"], summary["generated"]) == (4, 2, 0)
    assert summary["timing"] == {}


def test_no_generate_never_touches_the_store(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    before = store.read_text(encoding="utf-8")
    teacher.build_training_data(_train_rows(5), out, store, generate=False)
    assert store.read_text(encoding="utf-8") == before


def test_no_generate_with_nothing_stored_raises_and_writes_nothing(
    tmp_path, monkeypatch
):
    out, store = _paths(tmp_path)
    seen = _fake_generate_chain(monkeypatch)
    with pytest.raises(ValueError, match="no stored chains"):
        teacher.build_training_data(_train_rows(3), out, store, generate=False)
    assert seen == []
    assert not out.exists()
    assert not store.exists()


def test_no_generate_still_stops_for_a_changed_row(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    changed = _train_rows(3)
    changed[0] = {**changed[0], "question": "A different question entirely?"}
    with pytest.raises(ValueError, match="r-0"):
        teacher.build_training_data(changed, out, store, generate=False)


def test_no_generate_reports_failed_rows_and_skips_them(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch, fail_ids={"r-1"})
    teacher.build_training_data(_train_rows(3), out, store)
    summary = teacher.build_training_data(_train_rows(3), out, store, generate=False)
    assert summary["failed"] == [{"id": "r-1", "reason": "no_valid_chain"}]
    assert _ids(out) == ["r-0", "r-2"]


def test_a_normal_run_has_nothing_pending(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    assert teacher.build_training_data(_train_rows(3), out, store)["pending"] == 0


def test_no_generate_progress_still_ticks_once_per_row(tmp_path, monkeypatch):
    out, store = _paths(tmp_path)
    _fake_generate_chain(monkeypatch)
    teacher.build_training_data(_train_rows(2), out, store)
    ticks = []
    teacher.build_training_data(
        _train_rows(5), out, store, generate=False, progress=lambda: ticks.append(1)
    )
    assert len(ticks) == 5


def test_max_seq_length_is_just_above_the_longest_audited_example():
    # 296 tokens is the longest of the 383 audited chains; if later chains are
    # longer this fails on purpose: re-run `validate --tokens` and update both
    assert config.MAX_SEQ_LENGTH % 64 == 0
    assert 296 <= config.MAX_SEQ_LENGTH < 296 + 64


# ---- M8: planning helpers and config ----
def test_adapter_path_accepts_a_root_for_the_modal_volume(tmp_path):
    assert config.adapter_path(2, root=tmp_path) == tmp_path / "round_2"
    assert config.adapter_path(2) == config.ADAPTER_DIR / "round_2"


@pytest.mark.parametrize(
    ("n", "batch", "accum", "expected"),
    [
        (383, 4, 4, 24),
        (450, 4, 4, 29),
        (3, 4, 4, 1),
        (64, 4, 4, 4),
        (65, 4, 4, 5),
        (1, 1, 1, 1),
    ],
)
def test_steps_per_epoch_rounds_up_and_is_never_zero(n, batch, accum, expected):
    assert train.steps_per_epoch(n, batch, accum) == expected


def test_total_steps_is_steps_per_epoch_times_epochs():
    assert train.total_steps(383, 4, 4, 3) == 72


@pytest.mark.parametrize("args", [(0, 4, 4), (10, 0, 4), (10, 4, 0), (-1, 4, 4)])
def test_steps_per_epoch_rejects_non_positive_inputs(args):
    with pytest.raises(ValueError):
        train.steps_per_epoch(*args)


def test_the_training_hyperparameters_are_the_chosen_ones():
    assert config.PER_DEVICE_TRAIN_BATCH_SIZE == 4
    assert config.GRADIENT_ACCUMULATION_STEPS == 4
    assert config.EARLY_STOPPING is True
    assert config.EARLY_STOPPING_PATIENCE == 1
    assert config.EPOCHS == 3


def test_clean_overrides_drops_none_but_keeps_zero_and_false():
    cleaned = train.clean_overrides(
        {"seed": 0, "flag": False, "epochs": None, "lr": 1e-4}
    )
    assert cleaned == {"seed": 0, "flag": False, "lr": 1e-4}


def test_hyperparameters_cover_the_settings_that_change_a_run():
    h = train.hyperparameters()
    for key in (
        "model",
        "learning_rate",
        "epochs",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "lora_target_modules",
        "max_seq_length",
        "seed",
        "early_stopping",
        "early_stopping_patience",
    ):
        assert key in h
    assert h["learning_rate"] == config.LEARNING_RATE


def test_hyperparameter_hash_is_stable_and_changes_with_any_setting():
    base = train.hyperparameters()
    assert train.hyperparameter_hash(base) == train.hyperparameter_hash(
        train.hyperparameters()
    )
    assert train.hyperparameter_hash(base) != train.hyperparameter_hash(
        train.hyperparameters(learning_rate=1e-4)
    )
    assert train.hyperparameter_hash(base) != train.hyperparameter_hash(
        train.hyperparameters(epochs=5)
    )
    assert len(train.hyperparameter_hash(base)) == 64


def test_hyperparameter_overrides_must_name_a_known_setting():
    with pytest.raises(ValueError, match="learning_rte"):
        train.hyperparameters(learning_rte=1e-4)


# ---- M8: the text dataset ----
def test_build_text_dataset_formats_every_row_and_ends_with_the_end_token():
    rows = [_trow(1), _trow(2, chain=_teacher_chain("9", "other words"))]
    dataset = train.build_text_dataset(rows, config.EOS_TOKEN)
    assert list(dataset["text"]) == [
        data.format_example(r["question"], r["chain"], config.EOS_TOKEN) for r in rows
    ]
    assert all(text.endswith(config.EOS_TOKEN) for text in dataset["text"])


def test_build_text_dataset_rejects_an_empty_response_naming_the_row():
    with pytest.raises(ValueError, match="t-2"):
        train.build_text_dataset([_trow(1), _trow(2, chain="")], config.EOS_TOKEN)


def test_build_text_dataset_rejects_a_chain_with_a_missing_section_naming_the_row():
    broken = _without_line(_teacher_chain("9"), "Verify:")
    with pytest.raises(ValueError, match="t-1"):
        train.build_text_dataset([_trow(1, chain=broken)], config.EOS_TOKEN)


def test_build_text_dataset_with_no_rows_raises():
    with pytest.raises(ValueError, match="no rows"):
        train.build_text_dataset([], config.EOS_TOKEN)


# ---- M8: model loading and adapters (unsloth is faked) ----
def _fake_unsloth(monkeypatch):
    module = ModuleType("unsloth")
    fast = MagicMock()
    fast.from_pretrained.return_value = ("base_model", "tokenizer")
    fast.get_peft_model.return_value = "peft_model"
    module.FastLanguageModel = fast
    monkeypatch.setitem(sys.modules, "unsloth", module)
    return fast


def test_load_model_for_training_loads_the_base_model_in_4bit(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    train.load_model_for_training()
    kwargs = fast.from_pretrained.call_args.kwargs
    assert kwargs["model_name"] == config.MODEL_NAME
    assert kwargs["load_in_4bit"] is True
    assert kwargs["max_seq_length"] == config.MAX_SEQ_LENGTH


def test_load_model_for_training_attaches_lora_with_the_configured_settings(
    monkeypatch,
):
    fast = _fake_unsloth(monkeypatch)
    result = train.load_model_for_training()
    kwargs = fast.get_peft_model.call_args.kwargs
    assert fast.get_peft_model.call_args.args == ("base_model",)
    assert kwargs["r"] == config.LORA_R
    assert kwargs["lora_alpha"] == config.LORA_ALPHA
    assert kwargs["target_modules"] == config.LORA_TARGET_MODULES
    assert kwargs["lora_dropout"] == 0
    assert kwargs["bias"] == "none"
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == config.SEED
    assert result == ("peft_model", "tokenizer")


def test_load_model_for_training_does_not_switch_to_inference_mode(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    train.load_model_for_training()
    fast.for_inference.assert_not_called()


def test_save_adapter_saves_the_model_and_tokenizer():
    trainer, tokenizer = MagicMock(), MagicMock()
    train.save_adapter(trainer, tokenizer, "some/dir")
    trainer.save_model.assert_called_once_with("some/dir")
    tokenizer.save_pretrained.assert_called_once_with("some/dir")


def _write_adapter(root, round_n, base="unsloth/Llama-3.2-1B", rank=config.LORA_R):
    folder = config.adapter_path(round_n, root=root)
    folder.mkdir(parents=True)
    (folder / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "r": rank}), encoding="utf-8"
    )
    return folder


def test_load_adapter_returns_the_latest_rounds_folder(tmp_path):
    _write_adapter(tmp_path, 0)
    _write_adapter(tmp_path, 1)
    assert (
        train.load_adapter([{"round": 0}, {"round": 1}], root=tmp_path)
        == tmp_path / "round_1"
    )


def test_load_adapter_with_a_missing_folder_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="round_0"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_with_a_folder_but_no_config_raises(tmp_path):
    (tmp_path / "round_0").mkdir()
    with pytest.raises(FileNotFoundError, match="adapter_config.json"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_from_another_base_model_raises(tmp_path):
    _write_adapter(tmp_path, 0, base="meta-llama/Llama-3.1-8B")
    with pytest.raises(ValueError, match="base model"):
        train.load_adapter([{"round": 0}], root=tmp_path)


@pytest.mark.parametrize(
    "base",
    ["unsloth/Llama-3.2-1B", "unsloth/llama-3.2-1b-bnb-4bit", "UNSLOTH/LLAMA-3.2-1B"],
)
def test_load_adapter_accepts_the_same_base_model_under_its_usual_names(tmp_path, base):
    _write_adapter(tmp_path, 0, base=base)
    assert train.load_adapter([{"round": 0}], root=tmp_path) == tmp_path / "round_0"


def test_load_adapter_with_a_different_rank_raises(tmp_path):
    _write_adapter(tmp_path, 0, rank=config.LORA_R * 2)
    with pytest.raises(ValueError, match="rank"):
        train.load_adapter([{"round": 0}], root=tmp_path)


def test_load_adapter_with_a_gap_in_the_round_numbers_raises(tmp_path):
    _write_adapter(tmp_path, 0)
    _write_adapter(tmp_path, 2)
    with pytest.raises(ValueError, match="gap"):
        train.load_adapter([{"round": 0}, {"round": 2}], root=tmp_path)


def test_load_adapter_with_no_rounds_recorded_raises(tmp_path):
    with pytest.raises(ValueError, match="no rounds"):
        train.load_adapter([], root=tmp_path)


# ---- M8: the trainer (trl, transformers and unsloth are faked) ----
def _fake_training_stack(monkeypatch):
    trl = ModuleType("trl")
    trl.SFTConfig = MagicMock(name="SFTConfig")
    trl.SFTTrainer = MagicMock(name="SFTTrainer")
    transformers = ModuleType("transformers")
    transformers.EarlyStoppingCallback = MagicMock(name="EarlyStoppingCallback")
    chat = ModuleType("unsloth.chat_templates")
    chat.train_on_responses_only = MagicMock(
        name="train_on_responses_only", return_value="masked_trainer"
    )
    unsloth = ModuleType("unsloth")
    unsloth.chat_templates = chat
    for name, module in (
        ("trl", trl),
        ("transformers", transformers),
        ("unsloth", unsloth),
        ("unsloth.chat_templates", chat),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return trl, transformers, chat


def _datasets(n_train=3, n_eval=2):
    return ["train"] * n_train, ["eval"] * n_eval


def _trainer_with_fakes(monkeypatch, **overrides):
    trl, transformers, chat = _fake_training_stack(monkeypatch)
    train_ds, eval_ds = overrides.pop("datasets", _datasets())
    result = train.build_trainer("model", "tok", train_ds, eval_ds, "out", **overrides)
    return trl, transformers, chat, result


def test_build_trainer_configures_epochs_batching_and_evaluation_per_epoch(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    kwargs = trl.SFTConfig.call_args.kwargs
    assert kwargs["output_dir"] == "out"
    assert kwargs["num_train_epochs"] == config.EPOCHS
    assert kwargs["per_device_train_batch_size"] == config.PER_DEVICE_TRAIN_BATCH_SIZE
    assert kwargs["per_device_eval_batch_size"] == config.PER_DEVICE_EVAL_BATCH_SIZE
    assert kwargs["gradient_accumulation_steps"] == config.GRADIENT_ACCUMULATION_STEPS
    assert kwargs["learning_rate"] == config.LEARNING_RATE
    assert kwargs["eval_strategy"] == "epoch"
    assert kwargs["save_strategy"] == "epoch"
    assert kwargs["logging_strategy"] == "epoch"
    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False
    assert kwargs["dataset_text_field"] == "text"
    assert kwargs["max_length"] == config.MAX_SEQ_LENGTH
    assert kwargs["seed"] == config.SEED
    assert kwargs["report_to"] == "none"


def test_build_trainer_never_truncates_below_the_audited_limit(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    assert trl.SFTConfig.call_args.kwargs["max_length"] >= 296


def test_build_trainer_accepts_overrides_including_a_zero_seed(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(
        monkeypatch,
        epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        seed=0,
    )
    kwargs = trl.SFTConfig.call_args.kwargs
    assert (
        kwargs["num_train_epochs"],
        kwargs["per_device_train_batch_size"],
        kwargs["gradient_accumulation_steps"],
        kwargs["seed"],
    ) == (1, 2, 8, 0)


def test_build_trainer_passes_the_model_data_and_tokenizer(monkeypatch):
    trl, _, _, _ = _trainer_with_fakes(monkeypatch)
    kwargs = trl.SFTTrainer.call_args.kwargs
    assert kwargs["model"] == "model"
    assert kwargs["processing_class"] == "tok"
    assert kwargs["args"] is trl.SFTConfig.return_value
    assert len(kwargs["train_dataset"]) == 3 and len(kwargs["eval_dataset"]) == 2


def test_build_trainer_early_stopping_on_adds_one_callback_with_the_patience(
    monkeypatch,
):
    trl, transformers, _, _ = _trainer_with_fakes(
        monkeypatch, early_stopping=True, early_stopping_patience=2
    )
    transformers.EarlyStoppingCallback.assert_called_once_with(
        early_stopping_patience=2
    )
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == [
        transformers.EarlyStoppingCallback.return_value
    ]


def test_build_trainer_early_stopping_off_adds_no_stopping_callback(monkeypatch):
    trl, transformers, _, _ = _trainer_with_fakes(monkeypatch, early_stopping=False)
    transformers.EarlyStoppingCallback.assert_not_called()
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == []


def test_build_trainer_uses_the_configured_early_stopping_by_default(monkeypatch):
    _, transformers, _, _ = _trainer_with_fakes(monkeypatch)
    transformers.EarlyStoppingCallback.assert_called_once_with(
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )


def test_build_trainer_appends_extra_callbacks_after_early_stopping(monkeypatch):
    extra = object()
    trl, transformers, _, _ = _trainer_with_fakes(monkeypatch, extra_callbacks=[extra])
    assert trl.SFTTrainer.call_args.kwargs["callbacks"] == [
        transformers.EarlyStoppingCallback.return_value,
        extra,
    ]


def test_build_trainer_masks_everything_before_the_response(monkeypatch):
    trl, _, chat, result = _trainer_with_fakes(monkeypatch)
    chat.train_on_responses_only.assert_called_once_with(
        trl.SFTTrainer.return_value,
        instruction_part=config.INSTRUCTION_MARKER,
        response_part=config.RESPONSE_MARKER,
    )
    assert result == "masked_trainer"


def test_build_trainer_with_an_empty_eval_set_raises(monkeypatch):
    with pytest.raises(ValueError, match="eval dataset is empty"):
        _trainer_with_fakes(monkeypatch, datasets=(["t"], []))


def test_build_trainer_with_an_empty_train_set_raises(monkeypatch):
    with pytest.raises(ValueError, match="train dataset is empty"):
        _trainer_with_fakes(monkeypatch, datasets=([], ["e"]))


# ---- M8: the per-epoch evaluation ----
def _evaluator(outputs=None, raise_in_generate=False):
    rows = [
        _trow(1, answer="9", category="lateral_thinking"),
        _trow(2, answer="9", category="misconception"),
    ]
    outputs = outputs or [_teacher_chain("9"), _teacher_chain("12")]

    def generate(prompts):
        if raise_in_generate:
            raise RuntimeError("generation failed")
        return list(outputs)

    return train.EpochEvaluator(rows, generate)


def test_epoch_evaluator_records_the_epoch_losses_and_scores(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    evaluator = _evaluator()
    record = evaluator.evaluate_epoch("model", epoch=1, train_loss=1.5, eval_loss=1.2)
    assert record["epoch"] == 1
    assert (record["train_loss"], record["eval_loss"]) == (1.5, 1.2)
    assert record["accuracy"] == 0.5
    assert record["format_compliance"] == 1.0
    assert evaluator.epochs == [record]
    fast.for_inference.assert_called_once_with("model")
    fast.for_training.assert_called_once_with("model")


def test_epoch_evaluator_keeps_the_same_category_keys_every_epoch(monkeypatch):
    _fake_unsloth(monkeypatch)
    evaluator = _evaluator()
    first = evaluator.evaluate_epoch("m", 1, 1.0, 1.0)
    second = evaluator.evaluate_epoch("m", 2, 0.9, 0.9)
    assert (
        list(first["per_category"]) == list(second["per_category"]) == config.CATEGORIES
    )
    assert first["per_category"]["time_sensitive"] == "undefined"
    assert [e["epoch"] for e in evaluator.epochs] == [1, 2]


def test_epoch_evaluator_restores_training_mode_even_if_generation_throws(monkeypatch):
    fast = _fake_unsloth(monkeypatch)
    evaluator = _evaluator(raise_in_generate=True)
    with pytest.raises(RuntimeError, match="generation failed"):
        evaluator.evaluate_epoch("model", 1, 1.0, 1.0)
    fast.for_training.assert_called_once_with("model")
    assert evaluator.epochs == []


def test_epoch_evaluator_accepts_missing_losses(monkeypatch):
    _fake_unsloth(monkeypatch)
    record = _evaluator().evaluate_epoch("m", 1, train_loss=None, eval_loss=None)
    assert (record["train_loss"], record["eval_loss"]) == (None, None)


def test_epoch_evaluator_with_no_eval_rows_raises():
    with pytest.raises(ValueError, match="eval rows"):
        train.EpochEvaluator([], lambda prompts: [])


def _fake_transformers_callback(monkeypatch):
    module = ModuleType("transformers")

    class TrainerCallback:
        pass

    module.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", module)
    return TrainerCallback


def test_the_callback_runs_the_evaluator_at_each_evaluation_with_the_epoch_and_losses(
    monkeypatch,
):
    _fake_unsloth(monkeypatch)
    base = _fake_transformers_callback(monkeypatch)
    evaluator = _evaluator()
    callback = train.make_epoch_eval_callback(evaluator)
    assert isinstance(callback, base)
    state = SimpleNamespace(
        epoch=2.0,
        log_history=[{"loss": 1.9, "epoch": 1.0}, {"loss": 1.4, "epoch": 2.0}],
    )
    callback.on_evaluate(None, state, None, metrics={"eval_loss": 1.1}, model="model")
    (record,) = evaluator.epochs
    assert (record["epoch"], record["train_loss"], record["eval_loss"]) == (2, 1.4, 1.1)


def test_the_callback_handles_a_log_history_without_a_training_loss(monkeypatch):
    _fake_unsloth(monkeypatch)
    _fake_transformers_callback(monkeypatch)
    evaluator = _evaluator()
    state = SimpleNamespace(epoch=1.0, log_history=[])
    train.make_epoch_eval_callback(evaluator).on_evaluate(
        None, state, None, metrics={"eval_loss": 1.1}, model="m"
    )
    assert evaluator.epochs[0]["train_loss"] is None


# ---- M8: the training record and rounds.json ----
def _trainer_state(**overrides):
    base = {
        "epoch": 3.0,
        "global_step": 72,
        "best_model_checkpoint": "/checkpoints/round_0/checkpoint-48",
        "best_metric": 0.91,
        "log_history": [{"loss": 1.5, "epoch": 1.0}],
    }
    return SimpleNamespace(**{**base, **overrides})


def _record(**overrides):
    base = {
        "round_n": 0,
        "trainer_state": _trainer_state(),
        "train_rows": [_trow(1), _trow(2), _trow(3)],
        "eval_rows": [_trow(4)],
        "requested_epochs": 3,
        "epoch_results": [{"epoch": 1}],
        "versions": {"python": "3.11"},
        "train_seconds": 90.5,
        "gpu_summary": {"gpu_name": None},
        "peak_torch_memory_allocated_mib": None,
        "hyperparams": train.hyperparameters(),
    }
    return train.build_training_record(**{**base, **overrides})


def test_training_record_has_the_fields_and_counts_match_the_rows():
    record = _record()
    for field in (
        "round",
        "model_name",
        "train_examples",
        "eval_examples",
        "requested_epochs",
        "epochs_run",
        "steps_run",
        "stopped_early",
        "best_model_checkpoint",
        "best_metric",
        "train_seconds",
        "versions",
        "gpu_summary",
        "peak_torch_memory_allocated_mib",
        "log_history",
        "epoch_results",
        "hyperparameters",
        "hyperparameter_hash",
        "gpu_type",
    ):
        assert field in record, field
    assert (record["train_examples"], record["eval_examples"]) == (3, 1)
    assert record["round"] == 0
    assert record["best_model_checkpoint"] == "/checkpoints/round_0/checkpoint-48"
    assert record["best_metric"] == 0.91
    assert json.loads(json.dumps(record)) == record


def test_training_record_says_when_training_stopped_early():
    assert _record(trainer_state=_trainer_state(epoch=2.0))["stopped_early"] is True
    assert _record(trainer_state=_trainer_state(epoch=3.0))["stopped_early"] is False


def test_training_record_hash_matches_the_hyperparameters():
    record = _record()
    assert record["hyperparameter_hash"] == train.hyperparameter_hash(
        record["hyperparameters"]
    )


def test_training_record_allows_no_best_checkpoint():
    record = _record(
        trainer_state=_trainer_state(best_model_checkpoint=None, best_metric=None)
    )
    assert (record["best_model_checkpoint"], record["best_metric"]) == (None, None)


def test_read_rounds_of_a_missing_file_is_empty(tmp_path):
    assert train.read_rounds(tmp_path / "rounds.json") == []


def test_read_rounds_of_a_corrupt_file_raises_naming_it(tmp_path):
    path = tmp_path / "rounds.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="rounds.json"):
        train.read_rounds(path)


def test_read_rounds_of_something_that_is_not_a_list_raises(tmp_path):
    path = tmp_path / "rounds.json"
    path.write_text('{"round": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        train.read_rounds(path)


def test_record_round_appends_rounds_in_order(tmp_path):
    path = tmp_path / "results" / "rounds.json"
    train.record_round(path, {"round": 0, "x": 1})
    train.record_round(path, {"round": 1, "x": 2})
    assert [r["round"] for r in train.read_rounds(path)] == [0, 1]
    assert list(path.parent.iterdir()) == [path]


def test_record_round_refuses_a_round_that_is_already_recorded(tmp_path):
    path = tmp_path / "rounds.json"
    train.record_round(path, {"round": 0})
    with pytest.raises(ValueError, match="already"):
        train.record_round(path, {"round": 0})
    assert len(train.read_rounds(path)) == 1


@pytest.mark.parametrize(("existing", "new"), [([], 1), ([0], 2), ([0, 1], 3)])
def test_record_round_refuses_a_gap_in_the_round_numbers(tmp_path, existing, new):
    path = tmp_path / "rounds.json"
    for n in existing:
        train.record_round(path, {"round": n})
    with pytest.raises(ValueError, match="gap"):
        train.record_round(path, {"round": new})


# ---- M8: masking (the marker's tokenization and the labels Unsloth produces) ----
def _masked_example(tok, row, shift=0):
    full = tok.encode(
        data.format_example(row["question"], row["chain"], config.EOS_TOKEN),
        add_special_tokens=True,
    ).ids
    boundary = train.response_start_index(tok, row["question"]) + shift
    return {"input_ids": full, "labels": [-100] * boundary + full[boundary:]}


def test_response_start_index_is_the_length_of_the_prompt_including_the_start_token():
    tok = _tiny_tokenizer()
    assert train.response_start_index(tok, "a b") == 9


def test_expected_trained_text_is_the_response_after_the_marker_up_to_the_end_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b", chain=_teacher_chain("9"))
    text = train.expected_trained_text(tok, row)
    full = tok.encode(
        data.format_example(row["question"], row["chain"], config.EOS_TOKEN),
        add_special_tokens=True,
    ).ids
    assert text.rstrip().endswith(config.EOS_TOKEN)
    assert len(text.split()) == len(full) - train.response_start_index(tok, "a b")


def test_check_masking_accepts_labels_that_keep_only_the_response_and_end_token():
    tok = _tiny_tokenizer()
    rows = [_trow(1, question="a b"), _trow(2, question="c d e")]
    examples = [_masked_example(tok, r) for r in rows]
    assert train.check_masking(tok, rows, examples) is None


def test_check_masking_catches_an_off_by_one_that_leaves_the_marker_unmasked():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [_masked_example(tok, row, shift=-1)])


def test_check_masking_catches_an_off_by_one_that_masks_the_first_response_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [_masked_example(tok, row, shift=1)])


def test_check_masking_catches_a_masked_end_token():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    example = _masked_example(tok, row)
    example["labels"][-1] = -100
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [example])


def test_check_masking_catches_everything_masked():
    tok = _tiny_tokenizer()
    row = _trow(1, question="a b")
    example = _masked_example(tok, row)
    example["labels"] = [-100] * len(example["labels"])
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(tok, [row], [example])


def test_check_masking_with_different_counts_of_rows_and_examples_raises():
    tok = _tiny_tokenizer()
    with pytest.raises(ValueError, match="1 rows but 0"):
        train.check_masking(tok, [_trow(1)], [])


@pytest.mark.slow
def test_real_tokenizer_marker_boundary_has_no_off_by_one():
    tok = data.load_tokenizer(config.MODEL_NAME)
    for question, chain in (
        (LONG_Q, _teacher_chain("9")),
        ("Was Talos alive?", _teacher_chain("No", "Talos was a bronze automaton.")),
    ):
        row = _trow(1, question=question, chain=chain)
        assert train.expected_trained_text(tok, row) == chain + config.EOS_TOKEN
        prompt_ids = tok.encode(
            data.format_prompt(question), add_special_tokens=True
        ).ids
        assert tok.decode(prompt_ids[-4:], skip_special_tokens=False).endswith(
            config.RESPONSE_MARKER
        )


@pytest.mark.slow
def test_real_tokenizer_masked_labels_keep_exactly_the_response_and_end_token():
    tok = data.load_tokenizer(config.MODEL_NAME)
    rows = [
        _trow(1, question=LONG_Q, chain=_teacher_chain("9")),
        _trow(
            2,
            question="How do planes stay in the air?",
            chain=_teacher_chain("9", "wings"),
        ),
    ]
    examples = [_masked_example(tok, r) for r in rows]
    assert train.check_masking(tok, rows, examples) is None


# ---- both tokenizer APIs: `tokenizers` (encode gives .ids) and transformers (encode gives a list) ----
class _HFStyleTokenizer:
    """Behaves like the Hugging Face tokenizer Unsloth returns inside the container."""

    def __init__(self, inner):
        self._inner = inner

    def encode(self, text, add_special_tokens=True):
        return self._inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=True):
        return self._inner.decode(ids, skip_special_tokens=skip_special_tokens)


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


def test_masking_helpers_work_with_a_transformers_style_tokenizer():
    inner = _tiny_tokenizer()
    hf = _HFStyleTokenizer(inner)
    row = _trow(1, question="a b")
    assert train.response_start_index(hf, "a b") == train.response_start_index(
        inner, "a b"
    )
    assert train.expected_trained_text(hf, row) == train.expected_trained_text(
        inner, row
    )
    example = _masked_example(inner, row)
    assert train.check_masking(hf, [row], [example]) is None
    shifted = _masked_example(inner, row, shift=-1)
    with pytest.raises(ValueError, match="t-1"):
        train.check_masking(hf, [row], [shifted])


def test_check_prompt_length_works_with_a_transformers_style_tokenizer():
    hf = _HFStyleTokenizer(_tiny_tokenizer())
    assert evaluation.check_prompt_length(["a b c"], hf, 99) is None
    with pytest.raises(ValueError, match="1 prompt"):
        evaluation.check_prompt_length(["a b c d e f g h"], hf, 3)


def test_generate_fn_puts_the_padding_side_back_so_training_is_not_affected():
    tok = _FakeTokenizer(["x"])
    tok.padding_side = "right"
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.padding_side == "right"


def test_generate_fn_puts_the_padding_side_back_even_if_generation_throws():
    class _Boom(_FakeModel):
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    tok = _FakeTokenizer(["x"])
    tok.padding_side = "right"
    generate = evaluation.make_generate_fn(_Boom(), tok, max_new_tokens=5, batch_size=8)
    with pytest.raises(RuntimeError, match="generation failed"):
        generate(["p"])
    assert tok.padding_side == "right"


def test_generate_fn_restores_a_left_padding_side_too():
    tok = _FakeTokenizer(["x"])
    tok.padding_side = "left"
    evaluation.make_generate_fn(_FakeModel(), tok, max_new_tokens=5, batch_size=8)(
        ["p"]
    )
    assert tok.padding_side == "left"


# ---- M8: masking on the collated batch (padding-free packs examples into one row) ----
def _example_ids_and_labels(tok, row, shift=0):
    example = _masked_example(tok, row, shift=shift)
    return example["input_ids"], example["labels"]


def _packed_batch(tok, rows, shifts=None):
    shifts = shifts or [0] * len(rows)
    ids, labels = [], []
    for row, shift in zip(rows, shifts, strict=True):
        row_ids, row_labels = _example_ids_and_labels(tok, row, shift)
        ids += row_ids
        labels += row_labels
    return {"input_ids": [ids], "labels": [labels]}


def _padded_batch(tok, rows):
    pairs = [_example_ids_and_labels(tok, row) for row in rows]
    width = max(len(ids) for ids, _ in pairs)
    return {
        "input_ids": [ids + [0] * (width - len(ids)) for ids, _ in pairs],
        "labels": [labels + [-100] * (width - len(labels)) for _, labels in pairs],
    }


def _two_rows():
    return [
        _trow(1, question="a b", chain=_teacher_chain("9")),
        _trow(2, question="c d e f", chain=_teacher_chain("9", "extra words here")),
    ]


def test_unmasked_runs_splits_contiguous_trained_stretches():
    assert train.unmasked_runs([1, 2, 3, 4, 5, 6], [-100, 2, 3, -100, 5, 6]) == [
        [2, 3],
        [5, 6],
    ]
    assert train.unmasked_runs([1, 2], [-100, -100]) == []
    assert train.unmasked_runs([1, 2], [1, 2]) == [[1, 2]]


def test_check_batch_masking_accepts_a_packed_padding_free_batch():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(tok, rows, _packed_batch(tok, rows), expected_count=2)
        is None
    )


def test_check_batch_masking_accepts_a_padded_batch():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(tok, rows, _padded_batch(tok, rows), expected_count=2)
        is None
    )


def test_check_batch_masking_accepts_tensor_like_batches():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = {key: np.array(value) for key, value in _padded_batch(tok, rows).items()}
    assert train.check_batch_masking(tok, rows, batch) is None


def test_check_batch_masking_accepts_the_examples_in_either_order():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    assert train.check_batch_masking(tok, rows, _packed_batch(tok, rows[::-1])) is None


def test_check_batch_masking_catches_a_boundary_that_merges_two_examples():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    first_len = len(_example_ids_and_labels(tok, rows[0])[0])
    start = train.response_start_index(tok, rows[1]["question"])
    for i in range(
        first_len, first_len + start
    ):  # the second example's prompt left unmasked
        batch["labels"][0][i] = batch["input_ids"][0][i]
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(tok, rows, batch)


@pytest.mark.parametrize("shift", [-1, 1])
def test_check_batch_masking_catches_an_off_by_one_in_a_packed_batch(shift):
    tok = _tiny_tokenizer()
    rows = _two_rows()
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(
            tok, rows, _packed_batch(tok, rows, shifts=[0, shift])
        )


def test_check_batch_masking_catches_a_masked_end_token():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    batch["labels"][0][-1] = -100
    with pytest.raises(ValueError, match="does not match"):
        train.check_batch_masking(tok, rows, batch)


def test_check_batch_masking_with_nothing_trained_raises():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    batch = _packed_batch(tok, rows)
    batch["labels"] = [[-100] * len(batch["labels"][0])]
    with pytest.raises(ValueError, match="no trained tokens"):
        train.check_batch_masking(tok, rows, batch)


def test_check_batch_masking_checks_the_number_of_examples_when_it_is_known():
    tok = _tiny_tokenizer()
    rows = _two_rows()
    with pytest.raises(ValueError, match="1 trained"):
        train.check_batch_masking(
            tok, rows, _packed_batch(tok, rows[:1]), expected_count=2
        )


def test_check_batch_masking_works_with_a_transformers_style_tokenizer():
    inner = _tiny_tokenizer()
    rows = _two_rows()
    assert (
        train.check_batch_masking(
            _HFStyleTokenizer(inner), rows, _packed_batch(inner, rows)
        )
        is None
    )


@pytest.mark.slow
def test_real_tokenizer_packed_batch_masking_keeps_exactly_the_responses():
    tok = data.load_tokenizer(config.MODEL_NAME)
    rows = [
        _trow(1, question=LONG_Q, chain=_teacher_chain("9")),
        _trow(
            2,
            question="How do planes stay in the air?",
            chain=_teacher_chain("9", "wings"),
        ),
        _trow(
            3,
            question="What makes bread rise in the oven?",
            chain=_teacher_chain("9", "yeast"),
        ),
    ]
    assert (
        train.check_batch_masking(tok, rows, _packed_batch(tok, rows), expected_count=3)
        is None
    )
