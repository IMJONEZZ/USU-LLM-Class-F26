from src.prompts import format_example, format_prompt


def test_format_prompt_contains_instruction_and_unmodified_resume():
    resume = "  Python developer\n\nSkills:  SQL  "
    prompt = format_prompt(resume)
    assert "What job category does this resume belong to?" in prompt
    assert resume in prompt


def test_format_prompt_exact_layout():
    assert format_prompt("RESUME") == (
        "RESUME\n\n"
        "### Instruction:\n"
        "What job category does this resume belong to?\n\n"
        "### Response:\n"
    )


def test_format_example_is_prompt_followed_by_category():
    assert format_example("RESUME", "HR") == format_prompt("RESUME") + "HR"
