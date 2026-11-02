"""Writes answers.ipynb from the recorded training run.

The homework wants a notebook with the ten questions and the answers my fine
tuned model gave. The model itself only exists on a rented T4 for the length of
the run, so the notebook reads what that run printed rather than pretending it
can load a 1B on whatever machine opens it.

    uv run python scripts/build_answers_notebook.py

Then execute it so the outputs are real and not typed in by hand:

    uv run --with jupyter jupyter nbconvert --execute --inplace answers.ipynb
"""

import json
import pathlib

RUN = pathlib.Path("data/run.json")
NOTEBOOK = pathlib.Path("answers.ipynb")


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


def build(run):
    cells = [
        markdown(
            "# Assignment 6, the ten questions\n"
            "\n"
            "`src/trainer.py` fine tunes `meta-llama/Llama-3.2-1B-Instruct` on data built by\n"
            "`src/dataset.py`. None of the ten questions below are in the training set, and\n"
            "`src.dataset.check_leakage` fails the build if anything close to one of them\n"
            "turns up.\n"
            "\n"
            "The run happened on a T4 through Modal:\n"
            "\n"
            "```\n"
            "modal run gpu/paraphrase_app.py --copies 12\n"
            "modal run gpu/finetune_app.py --chat-rows 400\n"
            "```\n"
            "\n"
            "That wrote `data/run.json`, which is what this notebook reads. The answers below\n"
            "are what the model actually produced, with greedy decoding so they come out the\n"
            "same every time.\n"
        ),
        code(
            "import json\n"
            "\n"
            'with open("data/run.json") as handle:\n'
            "    run = json.load(handle)\n"
            "\n"
            "print(f\"gpu            {run['gpu']}\")\n"
            "print(f\"training rows  {run['rows']}\")\n"
            "print(f\"epochs         {run['epochs']}\")\n"
            "print(f\"final loss     {run['train_loss']:.3f}\")\n"
            "print(f\"train runtime  {run['train_runtime_s']:.0f}s\")\n"
            "print(f\"peak vram      {run['peak_vram_gb']} GB\")\n"
        ),
        markdown("## The ten questions, after fine tuning\n"),
        code(
            'for i, (q, a) in enumerate(zip(run["questions"], run["after"]), 1):\n'
            '    print(f"Q{i}. {q}")\n'
            '    print(f"    {a}")\n'
            "    print()\n"
        ),
        markdown(
            "## Before and after\n"
            "\n"
            "Same questions put to the stock checkpoint before any training, so it's clear\n"
            "which answers the fine tune is responsible for.\n"
        ),
        code(
            "for i, (q, before, after) in enumerate(\n"
            '    zip(run["questions"], run["before"], run["after"]), 1\n'
            "):\n"
            '    print(f"Q{i}. {q}")\n'
            '    print(f"  before: {before[:300]}")\n'
            '    print(f"  after:  {after[:300]}")\n'
            "    print()\n"
        ),
        markdown(
            "## Held out trick questions\n"
            "\n"
            "These come from the MisguidedAttention set and nothing here was trained on them.\n"
            "They're the check on whether the model learned the idea or just my phrasings.\n"
        ),
        code(
            'for i, (q, a) in enumerate(zip(run["held_out_questions"], run["held_out_after"]), 1):\n'
            '    print(f"{i}. {q[:110]}...")\n'
            '    print(f"   {a[:300]}")\n'
            "    print()\n"
        ),
        markdown("## What the training set was made of\n"),
        code(
            'for source, n in sorted(run["row_counts"].items(), key=lambda kv: -kv[1]):\n'
            '    print(f"{n:>5}  {source}")\n'
            "print(f\"{run['rows']:>5}  total\")\n"
        ),
    ]
    return {
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
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(json.loads(RUN.read_text())), indent=1) + "\n")
    print(f"wrote {NOTEBOOK}")
