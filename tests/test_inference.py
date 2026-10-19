import pytest

from src import config, inference

GOOD_CHAIN = (
    "Restate: A farmer has 17 sheep; all but 9 run away.\n"
    "Check: This is a wording trap.\n"
    "Solve: All but 9 means 9 remain.\n"
    "Verify: 17 - 8 = 9.\n"
    "Final answer: 9"
)


def _without(label):
    return "\n".join(
        line for line in GOOD_CHAIN.split("\n") if not line.startswith(label)
    )


def test_format_prompt_matches_the_a6_training_prompt():
    # copied from the A6 code's real output; the served model must see these exact characters
    expected = (
        "### Instruction:\nWhere did fortune cookies originate?\n\n### Response:\n"
    )
    assert inference.format_prompt("Where did fortune cookies originate?") == expected


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        f"What is {config.INSTRUCTION_MARKER.strip()} this?",
        f"What is {config.RESPONSE_MARKER.strip()} this?",
    ],
)
def test_format_prompt_rejects_empty_or_marker_questions(question):
    with pytest.raises(ValueError):
        inference.format_prompt(question)


def test_check_chain_accepts_a_well_formed_reply():
    inference.check_chain(GOOD_CHAIN)


@pytest.mark.parametrize(
    "chain",
    [
        *[_without(label) for label in config.SECTION_LABELS],
        GOOD_CHAIN + "\nSolve: again",
        GOOD_CHAIN.replace("Solve:", "Verify:", 1).replace("Verify: 17", "Solve: 17"),
        GOOD_CHAIN.replace("Final answer: 9", "Final answer:"),
        GOOD_CHAIN + f"\n{config.RESPONSE_MARKER.strip()}",
    ],
    ids=[
        *[f"missing {label}" for label in config.SECTION_LABELS],
        "duplicated Solve",
        "Solve and Verify swapped",
        "empty final answer",
        "contains a template marker",
    ],
)
def test_check_chain_rejects_a_malformed_reply(chain):
    with pytest.raises(ValueError):
        inference.check_chain(chain)
