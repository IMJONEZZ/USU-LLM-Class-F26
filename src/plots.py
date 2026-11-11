import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

from src import config, data

# dataviz palette slots 1-3 (validated for light surfaces); grey marks the dropped type
SERIES = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}
LABEL_INK = {"train": "#ffffff", "validation": "#0b0b0b", "test": "#0b0b0b"}
SURFACE, INK, MUTED, DROPPED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7ae"


def draw_split(counts, dropped, path):
    """Draw one stacked bar per type (train, validation, test counts) and save the chart to path."""
    # barh draws the first label at the bottom, so dropped types sit at the bottom and the largest kept type on top
    labels = [*dropped, *sorted(counts, key=lambda t: sum(counts[t].values()))]

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, name in enumerate(labels):
        if name in dropped:
            ax.barh(y, dropped[name], color=DROPPED, height=0.6)
            ax.text(
                dropped[name] + 0.3,
                y,
                f"{dropped[name]}  dropped (6 or fewer)",
                va="center",
                color=MUTED,
                fontsize=8,
            )
            continue
        left = 0
        for split, color in SERIES.items():
            n = counts[name][split]
            ax.barh(
                y,
                n,
                left=left,
                color=color,
                height=0.6,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=split if y == len(dropped) else None,
            )
            ax.text(
                left + n / 2,
                y,
                n,
                ha="center",
                va="center",
                color=LABEL_INK[split],
                fontsize=8,
            )
            left += n
        ax.text(left + 0.3, y, left, va="center", color=MUTED, fontsize=8)

    ax.set_yticks(range(len(labels)), labels, color=INK, fontsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # counts, so no 2.5 prompts
    ax.set_xlabel("prompts", color=MUTED)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"Every kept IFEval type reaches train, validation and test  ({sum(sum(c.values()) for c in counts.values())} prompts, {len(counts)} types)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def draw_pilot(record, path):
    """Draw how many tasks landed at each pass rate in the pilot, stacked by split, with the stop-rule line."""
    tasks = [t for t in record["tasks"] if t["rate"] is not None]
    rates = sorted({t["rate"] for t in tasks})
    totals = record["run"]["totals"]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for position, rate in enumerate(rates):
        bottom = 0
        for split in ("train", "validation"):
            n = sum(t["rate"] == rate and t["split"] == split for t in tasks)
            ax.bar(
                position,
                n,
                bottom=bottom,
                color=SERIES[split],
                width=0.6,
                edgecolor=SURFACE,
                linewidth=1.5,
                label=split if position == 0 else None,
            )
            if n:
                ax.text(
                    position,
                    bottom + n / 2,
                    n,
                    ha="center",
                    va="center",
                    color=LABEL_INK[split],
                    fontsize=9,
                )
            bottom += n
        ax.text(position, bottom + 1, bottom, ha="center", color=MUTED, fontsize=9)

    limit = config.STOP_ABOVE * len(tasks)
    ax.axhline(limit, color=MUTED, linestyle="--", linewidth=1)
    ax.text(
        len(rates) - 0.5,
        limit,
        f"stop rule: more than {limit:.0f} tasks ({config.STOP_ABOVE:.0%}) at 100%",
        ha="right",
        va="bottom",
        color=MUTED,
        fontsize=8,
    )
    ax.set_xticks(range(len(rates)), [f"{rate:.0%}" for rate in rates], color=INK)
    ax.set_xlabel(
        f"pass rate of a task over {record['run']['samples_per_task']} samples",
        color=MUTED,
    )
    ax.set_ylabel("tasks", color=MUTED)
    ax.tick_params(axis="y", colors=MUTED)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_title(
        f"One-line instruction, {len(tasks)} tasks: {totals['pass']} passes, {totals['fail']} fails, "
        f"{totals['parse_error']} parse errors, {totals['request_failure']} request failures",
        loc="left",
        color=INK,
        fontsize=10,
    )
    ax.legend(loc="center right", frameon=False, labelcolor=INK)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def draw_gepa(record, path):
    """Two charts: the validation score of each GEPA candidate, and what the run cost by model."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.5), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 2]}
    )
    scores = [c["validation_score"] for c in record["candidates"]]
    best = record["best_index"]
    left.set_facecolor(SURFACE)
    left.plot(
        range(len(scores)), scores, color=SERIES["train"], marker="o", linewidth=2
    )
    left.plot(
        best,
        scores[best],
        marker="o",
        markersize=11,
        color=SERIES["validation"],
        linestyle="",
        label="chosen",
    )
    for index, score in enumerate(scores):
        left.text(
            index, score + 0.012, f"{score:.0%}", ha="center", color=MUTED, fontsize=8
        )
    left.set_xticks(range(len(scores)))
    left.set_xlabel("GEPA candidate (0 is the one-line instruction)", color=MUTED)
    left.set_ylabel("validation pass rate (one sample per task)", color=MUTED)
    left.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    left.set_title(
        "Validation score of each candidate instruction",
        loc="left",
        color=INK,
        fontsize=10,
    )
    left.legend(frameon=False, labelcolor=INK, loc="lower right")

    right.set_facecolor(SURFACE)
    student, gemini = record["student"], record["gemini"]
    names = [
        f"Llama 3.2 3B on a T4\n{student['calls']} calls, {student['seconds']:.0f} s",
        f"Gemini\n{gemini['calls']} calls, {gemini['prompt_tokens'] + gemini['completion_tokens']:,} tokens",
    ]
    dollars = [student["dollars"], gemini["dollars"]]
    right.barh(
        names, dollars, color=[SERIES["train"], SERIES["validation"]], height=0.5
    )
    for position, value in enumerate(dollars):
        right.text(
            value, position, f" ${value:.2f}", va="center", color=INK, fontsize=9
        )
    right.set_xlim(0, max(dollars) * 1.3 or 1)
    right.set_title("What the run cost", loc="left", color=INK, fontsize=10)
    for ax in (left, right):
        ax.tick_params(colors=MUTED)
        ax.grid(axis="y" if ax is left else "x", color="#e4e3dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def draw_draws(rows, path):
    """Heatmap of the mean pass rate for each instruction type (rows) and instruction (columns), plus the overall rate per instruction."""
    names = ["one_liner", "baseline", "gepa"]
    types = sorted({r["instruction_id"] for r in rows})
    rates = {}  # (type, name) -> list of per-task pass rates
    for r in rows:
        graded = [d for d in r["draws"] if d["outcome"] != "request_failure"]
        if graded:
            rates.setdefault((r["instruction_id"], r["prompt"]), []).append(
                sum(d["outcome"] == "pass" for d in graded) / len(graded)
            )
    grid = [
        [
            sum(rates[t, n]) / len(rates[t, n]) if (t, n) in rates else float("nan")
            for n in names
        ]
        for t in types
    ]
    overall = [
        sum(sum(rates[t, n]) for t in types if (t, n) in rates)
        / sum(len(rates[t, n]) for t in types if (t, n) in rates)
        for n in names
    ]

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 9), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 1.3]}
    )
    cmap = LinearSegmentedColormap.from_list("blue", ["#eaf1fb", SERIES["train"]])
    left.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            left.text(
                x,
                y,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="#ffffff" if value > 0.6 else INK,
            )
    left.set_xticks(range(3), names, color=INK)
    left.set_yticks(range(len(types)), types, color=INK, fontsize=8)
    left.set_title(
        "Mean pass rate by instruction type (each cell averages its test tasks)",
        loc="left",
        color=INK,
        fontsize=10,
    )
    left.tick_params(length=0)

    right.set_facecolor(SURFACE)
    right.bar(
        names,
        overall,
        color=[SERIES["train"], SERIES["validation"], SERIES["test"]],
        width=0.6,
    )
    for x, value in enumerate(overall):
        right.text(x, value + 0.01, f"{value:.1%}", ha="center", color=INK, fontsize=10)
    right.set_ylim(0, 1)
    right.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    right.set_title("Overall (all test tasks)", loc="left", color=INK, fontsize=10)
    right.tick_params(colors=MUTED)
    right.grid(axis="y", color="#e4e3dd", linewidth=0.8)
    right.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        right.spines[side].set_visible(False)
        left.spines[side].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def draw_summary(summary, path):
    """Left: GEPA's gain over each other instruction with both bootstrap intervals. Right: the model's confidence against each task's pass rate."""
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(12, 4.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 2]}
    )
    comparisons = summary["comparisons"]
    left.set_facecolor(SURFACE)
    for y, c in enumerate(comparisons):
        left.plot(
            c["by_hand_interval"],
            [y + 0.12] * 2,
            color=SERIES["train"],
            linewidth=4,
            label="by hand" if y == 0 else None,
        )
        left.plot(
            c["scipy_interval"],
            [y - 0.12] * 2,
            color=SERIES["validation"],
            linewidth=4,
            label="SciPy" if y == 0 else None,
        )
        left.plot(c["mean_difference"], y, "o", color=INK, markersize=8)
        left.text(
            c["mean_difference"],
            y + 0.3,
            f"{c['mean_difference']:+.2f}",
            ha="center",
            color=INK,
            fontsize=9,
        )
    left.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    left.set_yticks(
        range(len(comparisons)),
        [f"gepa minus {c['from']}\n({c['tasks']} tasks)" for c in comparisons],
        color=INK,
    )
    left.set_ylim(-0.6, len(comparisons) - 0.2)
    left.set_xlabel(
        "mean change in a task's pass rate (95% bootstrap interval)", color=MUTED
    )
    left.set_title(
        "Does the optimized instruction help on the test tasks?",
        loc="left",
        color=INK,
        fontsize=10,
    )
    left.legend(frameon=False, labelcolor=INK, loc="lower right")

    right.set_facecolor(SURFACE)
    colors = {
        "one_liner": SERIES["train"],
        "baseline": SERIES["validation"],
        "gepa": SERIES["test"],
    }
    for name, color in colors.items():
        points = [p for p in summary["points"] if p["prompt"] == name]
        right.scatter(
            [p["avg_logprob"] for p in points],
            [p["rate"] for p in points],
            s=18,
            alpha=0.7,
            color=color,
            label=name,
            edgecolor=SURFACE,
        )
    rho = summary["spearman_logprob_vs_rate"]["all"]["rho"]
    right.set_xlabel(
        "average log-probability of the greedy answer (closer to 0 is more confident)",
        color=MUTED,
        fontsize=8,
    )
    right.set_ylabel("pass rate over the test draws", color=MUTED)
    right.set_title(
        f"Confidence against pass rate (Spearman rho {rho:+.2f})",
        loc="left",
        color=INK,
        fontsize=10,
    )
    right.legend(frameon=False, labelcolor=INK, fontsize=8)
    for ax in (left, right):
        ax.tick_params(colors=MUTED)
        ax.grid(color="#e4e3dd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


def main(figure="split"):
    if figure == "summary":
        summary = json.loads((Path(config.RESULTS_DIR) / "summary.json").read_text())
        draw_summary(summary, "results/figures/summary.png")
        return
    if figure == "gepa":
        record = json.loads((Path(config.RESULTS_DIR) / "gepa.json").read_text())
        draw_gepa(record, "results/figures/gepa.png")
        return
    if figure == "draws":
        lines = (Path(config.RESULTS_DIR) / "test_draws.jsonl").read_text().splitlines()
        draw_draws([json.loads(line) for line in lines], "results/figures/draws.png")
        return
    if figure == "pilot":
        record = json.loads(
            (Path(config.RESULTS_DIR) / "pilot_oneliner.json").read_text()
        )
        draw_pilot(record, "results/figures/pilot.png")
        return
    _, single = data.load_examples()
    kept, dropped = data.drop_small_types(single, config.MIN_TYPE_PROMPTS)
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    draw_split(data.split_counts(kept, splits), dropped, "results/figures/split.png")


if __name__ == "__main__":
    main(sys.argv[1])
