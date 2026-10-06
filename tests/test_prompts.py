from src.prompts import (
    INSTRUCTION_MARKER,
    RESPONSE_MARKER,
    format_example,
    format_prompt,
)

CATEGORIES = ["BPO", "HR", "INFORMATION-TECHNOLOGY"]


def test_format_prompt_contains_instruction_and_unmodified_resume():
    resume = "  Python developer\n\nSkills:  SQL  "
    prompt = format_prompt(resume, CATEGORIES)
    assert "What job category does this resume belong to?" in prompt
    assert resume in prompt


def test_format_prompt_lists_every_category_choice():
    prompt = format_prompt("RESUME", CATEGORIES)
    for category in CATEGORIES:
        assert category in prompt


def test_format_prompt_orders_categories_alphabetically():
    prompt = format_prompt("RESUME", ["INFORMATION-TECHNOLOGY", "HR", "BPO"])
    assert "Choose one of: BPO, HR, INFORMATION-TECHNOLOGY" in prompt


def test_format_prompt_exact_layout():
    assert format_prompt("RESUME", CATEGORIES) == (
        "RESUME\n\n"
        "### Instruction:\n"
        "What job category does this resume belong to? "
        "Choose one of: BPO, HR, INFORMATION-TECHNOLOGY\n\n"
        "### Response:\n"
    )


def test_markers_are_the_literal_template_strings():
    assert INSTRUCTION_MARKER == "### Instruction:\n"
    assert RESPONSE_MARKER == "### Response:\n"


def test_format_prompt_has_both_markers_in_order_and_ends_with_response():
    prompt = format_prompt("RESUME", CATEGORIES)
    assert prompt.index(INSTRUCTION_MARKER) < prompt.index(RESPONSE_MARKER)
    assert prompt.endswith(RESPONSE_MARKER)


def test_format_example_is_prompt_followed_by_category():
    assert format_example("RESUME", "HR", CATEGORIES) == (
        format_prompt("RESUME", CATEGORIES) + "HR"
    )
