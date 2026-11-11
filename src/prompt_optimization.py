"""Assignment 10: optimize a prompt with DSPy, without constrained sampling.

Task: "How many times does the letter 'l' appear in the word 'parallel'?"
Model: llama3.2 (3B) served by Ollama. The same model is used before and after.
Optimizer: BootstrapFewShot. It runs the model on training questions, keeps the
answers that were right, and adds them to the prompt as worked examples. It never
restricts which tokens the model may write, so there is no select or choice.

Run the real experiment (needs `ollama serve` and `ollama pull llama3.2`):

    uv run python -m src.prompt_optimization run
"""

import json
import random
import re
import sys
from pathlib import Path

import dspy

MODEL = "ollama_chat/llama3.2"
API_BASE = "http://localhost:11434/v1"

# 30 words with repeated letters, so counting is not trivial. Split by word, not by
# question, so no test word is ever seen while optimizing (no memorizing).
TRAIN_WORDS = [
    "strawberry",
    "mississippi",
    "banana",
    "committee",
    "bookkeeper",
    "assessment",
    "necessary",
    "occurrence",
    "balloon",
    "coffee",
    "different",
    "address",
    "success",
    "tomorrow",
    "possess",
    "embarrass",
    "millennium",
    "raspberry",
    "cappuccino",
    "unnecessary",
]
TEST_WORDS = [
    "parallel",
    "abbreviation",
    "accommodate",
    "pepper",
    "umbrella",
    "tattoo",
    "hippopotamus",
    "butterfly",
    "scissors",
    "giraffe",
]

TRAIN_SIZE = 40
VAL_SIZE = 20  # GEPA only: used to pick the best prompt; drawn from training words
TEST_SIZE = 60
GEPA_BUDGET = 200  # most model calls GEPA may spend scoring candidates
MAX_DEMOS = 4
SEED = 0

# The model often writes "twice" instead of "2", so the check must accept both.
NUMBER_WORDS = {
    "zero": "0",
    "no": "0",
    "once": "1",
    "one": "1",
    "twice": "2",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
}
_NUMBER_PATTERN = re.compile(r"\d+|\b(?:" + "|".join(NUMBER_WORDS) + r")\b")


def parse_count(text: str) -> str | None:
    """Return the last number in the text ("2", "twice", ...) as digits, or None."""
    found = _NUMBER_PATTERN.findall(str(text).lower())
    if not found:
        return None
    return NUMBER_WORDS.get(found[-1], found[-1])


def make_question(word: str, letter: str) -> str:
    return f"How many times does the letter '{letter}' appear in the word '{word}'?"


def make_examples(
    words: list[str],
    n: int,
    seed: int = SEED,
    exclude: list[dspy.Example] | None = None,
) -> list[dspy.Example]:
    """Build n distinct questions about the given words, with the true count as answer.

    Questions already in `exclude` are never used, so two sets can share words but not questions.
    """
    skip = {e.question for e in exclude or []}
    pairs = sorted(
        {(w, c) for w in words for c in set(w) if make_question(w, c) not in skip}
    )
    if n > len(pairs):
        raise ValueError(
            f"only {len(pairs)} distinct questions exist for these words, asked for {n}"
        )
    chosen = random.Random(seed).sample(pairs, n)
    return [
        dspy.Example(question=make_question(w, c), answer=str(w.count(c))).with_inputs(
            "question"
        )
        for w, c in chosen
    ]


def metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """True when the number in the model's answer equals the true count."""
    return parse_count(prediction.answer) == example.answer


def build_program() -> dspy.Predict:
    """The baseline: just question in, answer out. No hints."""
    return dspy.Predict("question -> answer")


def configure_lm(model: str = MODEL, api_base: str = API_BASE) -> dspy.LM:
    # cache=False so every run really calls the model; temperature 0 for repeatability.
    lm = dspy.LM(model, api_base=api_base, cache=False, temperature=0.0)
    dspy.configure(lm=lm)
    return lm


def evaluate(program: dspy.Module, devset: list[dspy.Example]) -> dict:
    """Score the program on the devset. Returns the accuracy and every answer."""
    result = dspy.Evaluate(
        devset=devset, metric=metric, num_threads=1, display_progress=False
    )(program)
    items = [
        {
            "question": example.question,
            "expected": example.answer,
            "model_answer": prediction.answer,
            "correct": bool(score),
        }
        for example, prediction, score in result.results
    ]
    return {
        "accuracy": result.score,
        "correct": sum(i["correct"] for i in items),
        "total": len(items),
        "items": items,
    }


def optimize(
    program: dspy.Module, trainset: list[dspy.Example], max_demos: int = MAX_DEMOS
) -> dspy.Module:
    """Run BootstrapFewShot: keep the model's own right answers as examples in the prompt."""
    optimizer = dspy.BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=max_demos, max_labeled_demos=0
    )
    return optimizer.compile(program, trainset=trainset)


def reference_scores(testset: list[dspy.Example]) -> dict:
    """Accuracy of lazy strategies that never read the question. Context for the real scores."""
    if not testset:
        return {}
    return {
        f"always_{label}": round(
            100 * sum(e.answer == digit for e in testset) / len(testset), 2
        )
        for label, digit in (("1", "1"), ("twice", "2"))
    }


def feedback_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """Score 1.0 or 0.0, plus written feedback that GEPA's reflection model reads."""
    if metric(example, prediction):
        return dspy.Prediction(
            score=1.0, feedback=f"Correct: the answer is {example.answer}."
        )
    letter, word = example.question.split("'")[1], example.question.split("'")[3]
    spots = [str(i + 1) for i, ch in enumerate(word) if ch == letter]
    feedback = (
        f"Wrong: the answer was {str(prediction.answer)!r} but the correct count is "
        f"{example.answer}. The word spelled out is {'-'.join(word)}. "
        f"The letter '{letter}' is at position(s) {', '.join(spots)}."
    )
    return dspy.Prediction(score=0.0, feedback=feedback)


def reflection_lm(model: str = MODEL, api_base: str = API_BASE) -> dspy.LM:
    """The model GEPA uses to rewrite the instructions. Here it is the same llama3.2."""
    return dspy.LM(
        model, api_base=api_base, cache=False, temperature=1.0, max_tokens=4000
    )


def make_gepa_optimizer(valset: list[dspy.Example], budget: int = GEPA_BUDGET):
    """Build an optimizer function that runs GEPA with a fixed budget of model calls."""

    def run(program: dspy.Module, trainset: list[dspy.Example]) -> dspy.Module:
        gepa = dspy.GEPA(
            metric=feedback_metric,
            reflection_lm=reflection_lm(),
            max_metric_calls=budget,
            num_threads=1,
            seed=SEED,
        )
        return gepa.compile(program, trainset=trainset, valset=valset)

    return run


def render_prompt(program: dspy.Module, question: str) -> str:
    """The exact text the model receives for this question, as plain text."""
    predictor = program.predictors()[0]
    prompt = dspy.ChatAdapter().format(
        predictor.signature, predictor.demos, {"question": question}
    )
    blocks = [f"[system]\n{prompt.system}"]
    for message in prompt.messages:
        text = "".join(part.text for part in message.parts)
        blocks.append(f"[{message.role}]\n{text}")
    return "\n\n".join(blocks)


def run_experiment(
    program: dspy.Module | None = None,
    trainset: list[dspy.Example] | None = None,
    testset: list[dspy.Example] | None = None,
    challenge_question: str | None = None,
    optimizer=None,
) -> dict:
    """Score the baseline, optimize, score again on the same unseen test questions.

    `optimizer` is any function (program, trainset) -> optimized program. The default is
    BootstrapFewShot.
    """
    baseline = program or build_program()
    trainset = (
        trainset if trainset is not None else make_examples(TRAIN_WORDS, TRAIN_SIZE)
    )
    testset = testset if testset is not None else make_examples(TEST_WORDS, TEST_SIZE)
    challenge = challenge_question or make_question("parallel", "l")

    before = evaluate(baseline, testset)
    optimized = (optimizer or optimize)(baseline, trainset)
    after = evaluate(optimized, testset)

    return {
        "model": MODEL,
        "train_size": len(trainset),
        "test_size": len(testset),
        "before": before,
        "after": after,
        "gain_points": round(after["accuracy"] - before["accuracy"], 2),
        "demos_in_prompt": len(optimized.predictors()[0].demos),
        "instructions_before": baseline.predictors()[0].signature.instructions,
        "instructions_after": optimized.predictors()[0].signature.instructions,
        "reference_scores": reference_scores(testset),
        "challenge_question": challenge,
        "prompt_before": render_prompt(baseline, challenge),
        "prompt_after": render_prompt(optimized, challenge),
        "answer_before": baseline(question=challenge).answer,
        "answer_after": optimized(question=challenge).answer,
    }


def program_with_instructions(instructions: str) -> dspy.Predict:
    """Rebuild an optimized program from the instruction text GEPA wrote (it adds no examples)."""
    program = build_program()
    predictor = program.predictors()[0]
    predictor.signature = predictor.signature.with_instructions(instructions)
    return program


def demo(question: str, results_path: str | Path) -> dict:
    """Ask the real model the same question with the baseline and the GEPA prompt."""
    saved = json.loads(Path(results_path).read_text())
    baseline = build_program()
    optimized = program_with_instructions(saved["instructions_after"])
    return {
        "question": question,
        "instructions_before": saved["instructions_before"],
        "instructions_after": saved["instructions_after"],
        "answer_before": baseline(question=question).answer,
        "answer_after": optimized(question=question).answer,
    }


def save_results(results: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2))


def main(argv: list[str]) -> int:
    if argv[:1] == ["run"]:
        configure_lm()
        results = run_experiment()
        out = "assignment_10/results/optimization_results.json"
    elif argv[:1] == ["run-gepa"]:
        configure_lm()
        trainset = make_examples(TRAIN_WORDS, TRAIN_SIZE)
        valset = make_examples(TRAIN_WORDS, VAL_SIZE, seed=SEED + 1, exclude=trainset)
        results = run_experiment(
            trainset=trainset, optimizer=make_gepa_optimizer(valset)
        )
        out = "assignment_10/results/gepa_results.json"
    elif argv[:1] == ["demo"] and len(argv) == 2:
        configure_lm()
        shown = demo(argv[1], "assignment_10/results/gepa_results.json")
        print(f"Question: {shown['question']}")
        print(f"\nBEFORE prompt instruction:\n  {shown['instructions_before']}")
        print(f"Answer: {shown['answer_before']}")
        print(f"\nAFTER prompt instruction:\n  {shown['instructions_after']}")
        print(f"Answer: {shown['answer_after']}")
        return 0
    else:
        print(
            'usage: python -m src.prompt_optimization run | run-gepa | demo "<question>"'
        )
        return 2
    save_results(results, out)
    print(
        f"before: {results['before']['accuracy']}%  after: {results['after']['accuracy']}%"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
