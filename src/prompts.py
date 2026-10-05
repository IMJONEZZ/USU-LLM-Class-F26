INSTRUCTION = "What job category does this resume belong to?"
RESPONSE_MARKER = "### Response:\n"


def format_prompt(resume_text):
    return f"{resume_text}\n\n### Instruction:\n{INSTRUCTION}\n\n{RESPONSE_MARKER}"


def format_example(resume_text, category):
    return format_prompt(resume_text) + category
