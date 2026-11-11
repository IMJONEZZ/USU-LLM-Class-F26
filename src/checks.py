from instruction_following_eval import evaluation_lib, instructions_registry
from langdetect import DetectorFactory

from src import config, data, prompts

# Google never seeds langdetect, so response_language could flip between runs
DetectorFactory.seed = 0


def passes(example, response):
    """True only when Google's strict check passes; an empty response fails."""
    official = evaluation_lib.InputExample(
        key=example["key"],
        instruction_id_list=[example["instruction_id"]],
        prompt=example["prompt"],
        kwargs=[example["kwargs"]],
    )
    result = evaluation_lib.test_instruction_following_strict(
        official, {example["prompt"]: response}
    )
    return result.follow_all_instructions


def describe(task):
    """Google's own description of the rule a task carries, built from the task's arguments."""
    instruction_id = task["instruction_id"]
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    return instruction.build_description(**task["kwargs"])


def feedback(example, response):
    """None when the response passes; otherwise the failed instruction's id and Google's own description of it."""
    if passes(example, response):
        return None
    return prompts.FEEDBACK_PROMPT.format(
        instruction_id=example["instruction_id"], description=describe(example)
    )


def main():
    _, single = data.load_examples()
    examples, _ = data.drop_small_types(single, config.MIN_TYPE_PROMPTS)

    # an empty response always fails, so every example must give feedback, the same text both times
    first = [feedback(e, "") for e in examples]
    same = sum(a == feedback(e, "") for a, e in zip(first, examples))
    named = sum(e["instruction_id"] in a for a, e in zip(first, examples))
    print(
        f"{len(examples)} prompts: same feedback on rebuild {same}, names its instruction {named}"
    )

    print("\nhand-check sample: first prompt of each type, empty response")
    seen = set()
    for e, text in zip(examples, first):
        if e["instruction_id"] not in seen:
            seen.add(e["instruction_id"])
            print(f"  kwargs {e['kwargs']}\n    {text}")

    print("\nthree failing responses and their feedback")
    demos = {
        "punctuation:no_comma": "Hello, world, this has commas.",
        "length_constraints:number_words": "Too short.",
        "change_case:english_lowercase": "Not Lowercase",
    }
    for e in examples:
        if e["instruction_id"] in demos:
            response = demos.pop(e["instruction_id"])
            print(
                f"  response {response!r} passes={passes(e, response)}\n    {feedback(e, response)}"
            )


if __name__ == "__main__":
    main()
