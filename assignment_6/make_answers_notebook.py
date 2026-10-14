"""Create an executed review notebook from actual saved GPU-generated answers."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.trainer import ASSIGNMENT_QUESTIONS


def build_notebook(answers_path: Path, training_path: Path, output_path: Path) -> None:
    payload = json.loads(answers_path.read_text())
    training = json.loads(training_path.read_text())
    if payload["questions"] != list(ASSIGNMENT_QUESTIONS):
        raise ValueError("saved outputs do not match the ten assignment questions")
    if len(payload["fine_tuned_answers"]) != 10 or not all(
        isinstance(a, str) and a.strip() for a in payload["fine_tuned_answers"]
    ):
        raise ValueError("need ten actual nonempty fine-tuned outputs")
    if payload.get("model_revision") != training.get("model_revision"):
        raise ValueError("answer and training model revisions differ")
    relative_path = answers_path.relative_to(output_path.parent)
    introduction = (
        "# Assignment 6: actual fine-tuned model answers\n\n"
        "These outputs were generated on Modal and saved before this notebook was built. "
        "The cells print those recorded outputs; they do not download or run the model.\n\n"
        f"Model: `{payload['model']}`  \nAdapter: `{payload['adapter']}`  \n"
        f"Pinned revision: `{payload['model_revision']}`  \n"
        f"Training settings: `{json.dumps(training['config'], sort_keys=True)}`\n\n"
        f"Generation settings: `{json.dumps(payload['settings'], sort_keys=True)}`\n\n"
        "The adapter was selected using development evidence. Final correctness scores "
        "under the prewritten guide are the student's decision."
    )
    selected = payload.get("selection")
    if selected:
        introduction += (
            "\n\nFinal development-selected checkpoint: "
            f"`{selected['selected_checkpoint']}`.  \n"
            f"Selection recorded: `{selected['selection_recorded_utc']}`.\n"
            "The training-result automatic best-checkpoint field is provisional; "
            "the final selection record identifies the model used for all ten answers."
        )
    decision_path = answers_path.parent / "final_selection.json"
    if decision_path.exists():
        decision = json.loads(decision_path.read_text())
        if decision["adapter"] != payload["adapter"]:
            raise ValueError("student decision refers to a different adapter")
        introduction += (
            "\n\nStudent accepted final score: **5/10**, "
            "correct questions **Q2, Q5, Q6, Q7, Q8**. "
            "Q1, Q3, Q4, Q9 and Q10 remain incorrect. "
            "The student chose any five correct answers as sufficient and accepted "
            "the Q1 regression. All ten printed outputs are from this one adapter; "
            "no answers were edited or combined."
        )
    cells = [{"cell_type": "markdown", "metadata": {}, "source": introduction}]
    sources = [
        f"import json\nfrom pathlib import Path\n\npayload = json.loads(Path({json.dumps(str(relative_path))}).read_text())"
    ]
    sources.extend(f'print(payload["fine_tuned_answers"][{i}])' for i in range(10))
    for i, source in enumerate(sources):
        if i:
            cells.append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": f"## Question {i}\n\n{payload['questions'][i - 1]}",
                }
            )
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            if i:
                print(payload["fine_tuned_answers"][i - 1])
        printed = stream.getvalue()
        if i and printed != payload["fine_tuned_answers"][i - 1] + "\n":
            raise ValueError("notebook output differs from the saved model answer")
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": i + 1,
                "source": source,
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": printed}
                ]
                if printed
                else [],
            }
        )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("answers.ipynb"))
    args = parser.parse_args()
    build_notebook(
        args.answers.resolve(), args.training_result.resolve(), args.output.resolve()
    )


if __name__ == "__main__":
    main()
