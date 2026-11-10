import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src import config, data

# dataviz palette slots 1-3 (validated for light surfaces); grey marks the dropped type
SERIES = {"train": "#2a78d6", "validation": "#eb6834", "test": "#1baf7a"}
LABEL_INK = {"train": "#ffffff", "validation": "#0b0b0b", "test": "#0b0b0b"}
SURFACE, INK, MUTED, DROPPED = "#fcfcfb", "#0b0b0b", "#52514e", "#b8b7ae"


def draw_split(
    path="results/figures/split.png",
):  # pragma: no cover - drawn output, checked by eye
    _, single = data.load_examples()
    kept, dropped = data.drop_small_types(single, config.MIN_TYPE_PROMPTS)
    splits = json.loads(Path(config.SPLITS_PATH).read_text())
    counts = data.split_counts(kept, splits)
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
        f"Every kept IFEval type reaches train, validation and test  ({sum(map(len, splits.values()))} prompts, {len(counts)} types)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower right", frameon=False, labelcolor=INK)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":  # pragma: no cover
    draw_split()
