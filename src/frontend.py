"""Assignment 11: a Gradio chat window for the Assignment 10 letter counter.

You type a question such as "How many times does the letter 'a' appear in the word
'banana'?". A radio switch under the chat box picks the prompt that answers it:
the plain prompt, or the prompt GEPA wrote in Assignment 10.

Run the window (needs `ollama serve` and `ollama pull llama3.2`):

    uv run python -m src.frontend serve

Score both prompts on the 60 Assignment 10 test questions, sent through the same reply
function the chat window calls:

    uv run python -m src.frontend measure
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path

import dspy
import gradio as gr

from src.prompt_optimization import (
    TEST_SIZE,
    TEST_WORDS,
    build_program,
    configure_lm,
    make_examples,
    metric,
    program_with_instructions,
)

PLAIN = "Plain prompt"
GEPA = "GEPA prompt"
CHOICES = [PLAIN, GEPA]
GEPA_RESULTS = "assignment_10/results/gepa_results.json"
FRONTEND_RESULTS = "assignment_11/results/frontend_results.json"
EMPTY_MESSAGE = "Ask a question, for example: How many times does the letter 'a' appear in the word 'banana'?"

Reply = Callable[[str, list, str], str]


def load_programs(results_path: str | Path = GEPA_RESULTS) -> dict[str, dspy.Module]:
    """The two prompts: the plain baseline, and the instruction GEPA wrote (saved in A10)."""
    saved = json.loads(Path(results_path).read_text())
    return {
        PLAIN: build_program(),
        GEPA: program_with_instructions(saved["instructions_after"]),
    }


def make_reply(programs: dict[str, dspy.Module]) -> Reply:
    """Build the function the chat window calls: (message, history, prompt choice) -> answer."""

    def reply(message: str, history: list, prompt_choice: str) -> str:
        if not message or not message.strip():
            return EMPTY_MESSAGE
        program = programs.get(prompt_choice)
        if program is None:
            return f"Unknown prompt choice {prompt_choice!r}. Pick one of: {', '.join(CHOICES)}."
        return str(program(question=message.strip()).answer)

    return reply


def build_app(reply: Reply) -> gr.ChatInterface:
    """The chat window: a ChatInterface plus a radio switch for the prompt."""
    return gr.ChatInterface(
        fn=reply,
        additional_inputs=[gr.Radio(CHOICES, value=PLAIN, label="Prompt")],
        title="Letter counter",
        description="Ask how many times a letter appears in a word. Pick which prompt answers.",
    )


def score_through_reply(
    reply: Reply, testset: list[dspy.Example], prompt_choice: str
) -> dict:
    """Send every test question through the reply function, then count the right answers."""
    items = []
    for example in testset:
        answer = reply(example.question, [], prompt_choice)
        correct = metric(example, dspy.Prediction(answer=answer))
        items.append(
            {
                "question": example.question,
                "expected": example.answer,
                "model_answer": answer,
                "correct": correct,
            }
        )
    right = sum(i["correct"] for i in items)
    return {
        "accuracy": round(100 * right / len(items), 2) if items else 0.0,
        "correct": right,
        "total": len(items),
        "items": items,
    }


def measure(reply: Reply, testset: list[dspy.Example] | None = None) -> dict:
    """Score both prompts and check the pass bar written in the plan before the run."""
    testset = testset if testset is not None else make_examples(TEST_WORDS, TEST_SIZE)
    plain = score_through_reply(reply, testset, PLAIN)
    gepa = score_through_reply(reply, testset, GEPA)
    gain = round(gepa["accuracy"] - plain["accuracy"], 2)
    bar = {
        "plain_within_5_of_a10": abs(plain["accuracy"] - 20.0) <= 5,
        "gepa_within_5_of_a10": abs(gepa["accuracy"] - 41.67) <= 5,
        "gain_at_least_15": gain >= 15,
    }
    return {
        "test_size": len(testset),
        "plain": plain,
        "gepa": gepa,
        "gain_points": gain,
        "pass_bar": bar,
        "passed": all(bar.values()),
    }


def save_results(results: dict, path: str | Path = FRONTEND_RESULTS) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2))


def main(argv: list[str]) -> int:
    if argv[:1] not in (["serve"], ["measure"]):
        print("usage: python -m src.frontend serve | measure")
        return 2
    configure_lm()
    reply = make_reply(load_programs())
    if argv[0] == "serve":  # pragma: no cover
        build_app(reply).launch()
        return 0
    results = measure(reply)
    save_results(results)
    print(
        f"plain: {results['plain']['accuracy']}%  gepa: {results['gepa']['accuracy']}%  "
        f"gain: {results['gain_points']}  passed: {results['passed']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
