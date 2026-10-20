import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src import config

SEGMENTS = [
    "container boot",
    "download",
    "weights load",
    "compile + graph capture",
    "profile + warmup",
    "other vLLM",
    "routing (our polling)",
    "first request",
]
# the dataviz skill's validated categorical order (light surface); slot i colors SEGMENTS[i]
COLORS = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def segments(record):
    """Non-overlapping pieces of one cold start. Steps vLLM did not print stay out and
    fall into 'other vLLM' instead of being guessed. Assumes download time is inside
    vLLM's 'Model loading took' and compile/graph time is inside 'init engine'."""
    download = record["vllm_download_s"] or 0
    weights = max((record["vllm_model_load_s"] or 0) - download, 0)
    compiled = (record["vllm_torch_compile_s"] or 0) + (
        record["vllm_graph_capture_s"] or 0
    )
    profile = max((record["vllm_init_engine_s"] or 0) - compiled, 0)
    other = max(record["vllm_startup"] - download - weights - compiled - profile, 0)
    return {
        "container boot": record["container_boot"] + record["imports"],
        "download": download,
        "weights load": weights,
        "compile + graph capture": compiled,
        "profile + warmup": profile,
        "other vLLM": other,
        "routing (our polling)": record["routing_gap"],
        "first request": record["first_request"],
    }


def run_label(record, name):
    fast = "on" if record["fast_boot"] else "off"
    return (
        f"{name}\nFAST_BOOT {fast} · HF cache {record['hf_cache_files']} files "
        f"· vLLM cache {record['vllm_cache_files']} files"
    )


def text_on(color):
    red, green, blue = (int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * red + 0.7152 * green + 0.0722 * blue > 0.45 else "white"


def draw(records, labels, out_path):
    plt.switch_backend("Agg")
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(records) + 2.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for row, record in enumerate(records):
        left = 0.0
        for color, seconds in zip(COLORS, segments(record).values()):
            ax.barh(row, seconds, left=left, height=0.5, color=color, edgecolor=SURFACE)
            if seconds >= 12:
                ax.text(
                    left + seconds / 2,
                    row,
                    f"{seconds:.0f}",
                    ha="center",
                    va="center",
                    color=text_on(color),
                    fontsize=9,
                )
            left += seconds
        ax.text(left + 2, row, f"{left:.0f} s", va="center", color=INK, fontsize=10)
    ax.set_yticks(range(len(records)), labels=labels, color=INK, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(
        "seconds from request sent to first token (one cold start per run)",
        color=INK_SOFT,
    )
    ax.set_title("Where a cold start spends its time", loc="left", color=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_SOFT, length=0)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    handles = [
        Patch(facecolor=color, edgecolor=SURFACE, label=name)
        for color, name in zip(COLORS, SEGMENTS)
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        frameon=False,
        labelcolor=INK_SOFT,
        fontsize=9,
    )
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def print_table(records, names):
    print(f"{'seconds':26}" + "".join(f"{name:>12}" for name in names))
    for label in SEGMENTS:
        values = "".join(f"{segments(r)[label]:12.1f}" for r in records)
        print(f"{label:26}{values}")
    print(f"{'total':26}" + "".join(f"{r['ttft_s']:12.1f}" for r in records))


def main(names):
    paths = [Path(config.RUNS_DIR) / f"{name}.json" for name in names]
    records = [json.loads(path.read_text())[0] for path in paths]
    labels = [run_label(record, name) for record, name in zip(records, names)]
    out_path = Path(config.RUNS_DIR) / "m8_chart.png"
    draw(records, labels, out_path)
    print_table(records, names)
    print("chart:", out_path)


if __name__ == "__main__":
    main(sys.argv[1:])
