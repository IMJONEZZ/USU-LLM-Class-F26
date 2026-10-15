"""Produce a reviewable results draft only from matching completed experiments."""

import argparse
import json
from pathlib import Path


def render_report(before, after, training):
    if before.get("status") != "complete" or after.get("status") != "complete":
        raise ValueError("Both evaluations must have completed scoring")
    for key in (
        "signature",
        "model",
        "revision",
        "split",
        "batch_size",
        "device",
        "max_new_tokens",
        "generation",
        "normalization",
        "empty_treatment",
    ):
        if before["metadata"][key] != after["metadata"][key]:
            raise ValueError(f"Before/after mismatch: {key}")
    if before["metadata"]["adapter"] or not after["metadata"]["adapter"]:
        raise ValueError("Compare an unmodified baseline against a trained adapter")
    if training["metadata"]["signature"] != before["metadata"]["signature"]:
        raise ValueError("Training belongs to another experiment")
    expected_run = {
        **training["metadata"],
        "checkpoint_epoch": training["best_epoch"],
        "checkpoint_dev_loss": training["best_dev_loss"],
    }
    if after["metadata"]["training_run"] != expected_run:
        raise ValueError(
            "Final adapter is not the selected checkpoint from this training run"
        )
    fields = ("example_id", "prompt", "reference")
    if [tuple(r[k] for k in fields) for r in before["examples"]] != [
        tuple(r[k] for k in fields) for r in after["examples"]
    ]:
        raise ValueError("Evaluation examples differ")

    def values(report):
        m = report["metrics"]
        return {
            "Exact-span match": m["exact_span_match"],
            "ROUGE-1": m["rouge"]["rouge1"],
            "ROUGE-L": m["rouge"]["rougeL"],
            "BLEU": m["bleu"]["bleu"],
            **{f"BERTScore {k}": v for k, v in m["bertscore"].items()},
        }

    a, b = values(before), values(after)
    lines = [
        "# Question 2 — measured results draft",
        "",
        "| Metric | Before | After | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in a:
        lines.append(
            f"| {key} | {a[key]:.6f} | {b[key]:.6f} | {b[key] - a[key]:+.6f} |"
        )
    delta = b["Exact-span match"] - a["Exact-span match"]
    direction = (
        "improved" if delta > 0 else "worsened" if delta < 0 else "stayed the same"
    )
    lines += [
        "",
        (
            f"Held-out exact-span match {direction}; inspect the other metrics separately. "
            "This is a descriptive comparison, not a statistical significance claim."
        ),
        "",
        f"Training took {training['training_seconds']:.3f} seconds. Scope: {training['timing_scope']}.",
        f"Peak allocated bytes: {training['peak_allocated_bytes']}; peak reserved bytes: {training['peak_reserved_bytes']}.",
        (
            f"Initial development loss: {training['initial_dev_loss']:.6f}. "
            f"Selected epoch: {training['best_epoch']}, development loss: {training['best_dev_loss']:.6f}."
        ),
        "",
        "| Epoch | Training loss | Development loss |",
        "| --- | ---: | ---: |",
    ]
    for row in training["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | {row['dev_loss']:.6f} |"
        )
    lines += [
        "",
        (
            "Loss is mean next-token cross-entropy over answer tokens and EOS. Lower loss means "
            "greater probability assigned to those targets. Training loss alone does not establish "
            "held-out improvement: memorization can reduce it, and teacher-forced loss differs from greedy generation."
        ),
        "",
        "Examples below are the first two successes and first two failures in fixed ID order, not a random sample.",
    ]
    for success in (True, False):
        chosen = [r for r in after["examples"] if r["exact"] == success][:2]
        if not chosen:
            lines.append(
                f"\nNo {'successes' if success else 'failures'} in this evaluation."
            )
        for row in chosen:
            old = next(
                r for r in before["examples"] if r["example_id"] == row["example_id"]
            )
            lines += [
                "",
                f"Example {row['example_id']} ({'success' if success else 'failure'}):",
                "```json",
                json.dumps(
                    {
                        "reference": row["reference"],
                        "before": old["prediction"],
                        "after": row["prediction"],
                    },
                    ensure_ascii=False,
                ),
                "```",
            ]
    lines += [
        "",
        (
            "Possible additional data (not used here): "
            "[DailyDialog](https://aclanthology.org/I17-1099/) contains everyday conversations; "
            "applying the same span task could broaden conversational phrasing. "
            "[Cornell Movie-Dialogs](https://convokit.cornell.edu/documentation/movie.html) could add "
            "script-style dialogue. These are hypotheses, not measured gains. Split by conversation/movie "
            "and remove overlap with held-out Star Wars dialogue before using either."
        ),
        "",
        "Review draft only. No PR comment or original Action Plan was edited.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = render_report(
        *(
            json.loads(path.read_text())
            for path in (args.before, args.after, args.training)
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        file.write(report)


if __name__ == "__main__":
    main()
