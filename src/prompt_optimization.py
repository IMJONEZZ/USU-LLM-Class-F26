"""
Use DSPy to optimize a prompt for speaker attribution: given a line of
Star Wars dialogue with the speaker's name removed, guess which character
said it. Originally planned to use MIPROv2 (which rewrites instructions,
not just few-shot examples), but MIPROv2 proved impractical given Groq's
free-tier rate limits (30 requests/minute) -- a single MIPROv2 "light" run
makes hundreds of calls across many trials, which pushed per-call latency
into multi-minute territory once rate-limited. Switched to BootstrapFewShot
instead, which only needed ~15-40 calls to find good few-shot examples.
"""

import os
import random
import unicodedata

import dspy

from src.dataloader import load_scenes

DATA_PATH = "data/star_wars_script.jsonl"


def extract_labeled_lines(data_path=DATA_PATH, max_lines=300, min_length=15):
    """
    Load the Star Wars dataset and extract individual dialogue lines paired
    with their speaker, filtering out very short lines (like "Yes" or "I
    know") that don't carry enough signal for speaker attribution to be a
    meaningful task.
    """
    scenes = load_scenes(data_path)
    lines = []
    for scene in scenes:
        for turn in scene["turns"]:
            if len(turn["text"]) >= min_length:
                lines.append({"speaker": turn["speaker"], "text": turn["text"]})
            if len(lines) >= max_lines:
                return lines
    return lines


def split_dataset(lines, train_frac=0.6, dev_frac=0.2, seed=17):
    """
    Split lines into train/dev/test sets. Shuffled with a fixed seed for
    reproducibility, since the order lines appear in the source file isn't
    randomized.
    """
    shuffled = lines.copy()
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)

    train = shuffled[:train_end]
    dev = shuffled[train_end:dev_end]
    test = shuffled[dev_end:]

    return train, dev, test


class SpeakerAttribution(dspy.Signature):
    """Given a line of dialogue, identify which character said it."""

    # This bare instruction, with zero mention of Star Wars, is deliberate:
    # it's the "before" baseline. The model has no context about which
    # universe this dialogue is from, which turned out to be the single
    # biggest driver of wrong guesses (e.g. "Captain Kirk", "Homer Simpson",
    # "Rick Grimes" -- all wrong-franchise guesses, not just wrong characters).
    dialogue_line: str = dspy.InputField(
        desc="A line of dialogue with the speaker removed"
    )
    speaker: str = dspy.OutputField(desc="The name of the character who said this line")


def exact_match_metric(example, prediction, trace=None):
    """
    Score 1.0 if the predicted speaker matches the actual speaker, else 0.0.
    Normalizes Unicode (e.g. accented characters) so encoding differences
    don't cause false mismatches -- caught a real bug where "Padmé" vs
    "Padme" was being scored as a mismatch purely due to a display/encoding
    issue, not an actual wrong answer.
    """

    def normalize(text):
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
            .lower()
        )

    predicted = normalize(prediction.speaker)
    actual = normalize(example.speaker)
    return 1.0 if predicted == actual else 0.0


def build_dspy_examples(lines):
    """Convert plain dicts into dspy.Example objects, with 'dialogue_line' as input."""
    examples = []
    for line in lines:
        example = dspy.Example(
            dialogue_line=line["text"],
            speaker=line["speaker"],
        ).with_inputs("dialogue_line")
        examples.append(example)
    return examples


def evaluate(predictor, dataset, metric):  # pragma: no cover
    """
    Run a predictor against a full dataset and return the average metric
    score. Requires real API calls, so this isn't tested in CI.
    """
    scores = []
    for example in dataset:
        try:
            prediction = predictor(dialogue_line=example["text"])
            example_obj = dspy.Example(speaker=example["speaker"])
            score = metric(example_obj, prediction)
        except Exception:  # noqa: BLE001 -- deliberately broad: any failure during evaluation scores as 0
            score = 0.0
        scores.append(score)
    return sum(scores) / len(scores)


def optimize_prompt(
    predictor, train_examples, metric, auto="light"
):  # pragma: no cover
    """
    Run MIPROv2 to optimize the predictor's instructions and few-shot
    examples against the training set, using the given metric. Not used in
    the final run (see module docstring) -- kept here since it's correct,
    standard DSPy usage, just impractical under this project's rate limits.
    """
    from dspy.teleprompt import MIPROv2

    optimizer = MIPROv2(metric=metric, auto=auto)
    optimized = optimizer.compile(predictor, trainset=train_examples)
    return optimized


if __name__ == "__main__":  # pragma: no cover
    # reasoning_effort="low" and a generous max_tokens were both necessary:
    # this model shows its own reasoning chain before answering, and without
    # these settings, responses were frequently truncated mid-reasoning,
    # before ever reaching the actual answer.
    lm = dspy.LM(
        "groq/openai/gpt-oss-20b",
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=1000,
        reasoning_effort="low",
    )
    dspy.configure(lm=lm)

    lines = extract_labeled_lines()
    print(f"Loaded {len(lines)} labeled lines")

    train, dev, test = split_dataset(lines)
    print(f"Train: {len(train)}, Dev: {len(dev)}, Test: {len(test)}")

    predictor = dspy.Predict(SpeakerAttribution)

    print("\nBaseline predictions (unoptimized):")
    for example in test[:5]:
        result = predictor(dialogue_line=example["text"])
        print(f"Line: {example['text']}")
        print(f"Actual: {example['speaker']} | Predicted: {result.speaker}")
        print("-" * 60)
