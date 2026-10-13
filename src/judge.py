import json

from tqdm import tqdm

from src import config, teacher


def judge_prompt(row, response):
    alternatives = ", ".join(row["accepted_alternatives"])
    return (
        "Grade one response to a test question using only the answer key. Do not "
        "use your own knowledge of the world.\n"
        f"Question: {row['question']}\n"
        f"Answer key: {row['answer']}\n"
        f"Other accepted wordings: {alternatives}\n"
        f"Response to grade: {response}\n"
        "correct: true only if the response gives the same answer as the key, even "
        "if it is worded differently or has extra words; false otherwise.\n"
        'Reply with JSON only, like {"correct": true}.'
    )


def judge_answer(row, response):
    text = teacher._generate(
        judge_prompt(row, response), config.JUDGE_MODEL, config.JUDGE_TEMPERATURE
    )
    correct = json.loads(teacher._unwrap(text))["correct"]
    if not isinstance(correct, bool):
        raise TypeError(f"judge reply is not a yes or no: {text!r}")
    return correct


def judge_all(rows, graded):
    if len(rows) != len(graded):
        raise ValueError(f"{len(rows)} rows but {len(graded)} outputs")
    return [
        judge_answer(row, output["final_answer"])
        for row, output in tqdm(list(zip(rows, graded, strict=True)))
    ]


def judged_metrics(graded, verdicts):
    if len(graded) != len(verdicts):
        raise ValueError(f"{len(graded)} rows but {len(verdicts)} verdicts")
    per_category = {}
    for category in config.CATEGORIES:
        mine = [
            v
            for g, v in zip(graded, verdicts, strict=True)
            if g["category"] == category
        ]
        per_category[category] = {
            "n": len(mine),
            "correct": sum(mine),
            "accuracy": sum(mine) / len(mine) if mine else "undefined",
        }
    return {
        "n": len(graded),
        "accuracy": sum(verdicts) / len(verdicts) if verdicts else "undefined",
        "per_category": per_category,
    }
