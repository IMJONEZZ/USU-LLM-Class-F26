INSTRUCTION = "What job category does this resume belong to?"
RESPONSE_MARKER = "### Response:\n"


def format_prompt(resume_text, categories):
    choices = ", ".join(sorted(categories))
    return (
        f"{resume_text}\n\n### Instruction:\n{INSTRUCTION} "
        f"Choose one of: {choices}\n\n{RESPONSE_MARKER}"
    )


def format_example(resume_text, category, categories):
    return format_prompt(resume_text, categories) + category
