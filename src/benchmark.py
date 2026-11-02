import json
import statistics
import sys
from collections import Counter
from html import escape
from pathlib import Path

METHODS = [
    "guidance",
    "coercion",
]  # order of calls within each prompt: Method A, then Method B


def call(generator, method, request):
    return getattr(generator, f"run_{method}").remote(request)


def run_benchmark(generator, prompts):
    """One discarded warm-up per method that is still cold in this container, then A and B on every prompt."""
    warm = generator.warm_state.remote()
    warmups = [call(generator, m, prompts[0]) for m in METHODS if warm.get(m, 0) == 0]
    records = []
    for prompt_id, prompt in enumerate(prompts):
        for method in METHODS:
            records.append(
                {
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    **call(generator, method, prompt),
                }
            )
    return {"warmups": warmups, "records": records}


def summarize(records):
    summary = {}
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        buckets = Counter(r["bucket"] for r in rows)
        seconds = [r["seconds"] for r in rows]
        tokens = [r["new_tokens"] for r in rows]
        summary[method] = {
            "n": len(rows),
            "valid": buckets["valid"],
            "malformed": buckets["malformed"],
            "truncated": buckets["truncated"],
            "valid_rate": round(buckets["valid"] / len(rows), 2),
            "mean_seconds": round(statistics.mean(seconds), 2),
            "median_seconds": round(statistics.median(seconds), 2),
            "mean_tokens": round(statistics.mean(tokens), 2),
            "seconds_per_token": round(sum(seconds) / sum(tokens), 2),
        }
    return summary


def format_table(summary):
    columns = list(next(iter(summary.values())))
    lines = ["method".ljust(10) + "".join(c.rjust(18) for c in columns)]
    for method, row in summary.items():
        lines.append(method.ljust(10) + "".join(str(row[c]).rjust(18) for c in columns))
    return "\n".join(lines)


def strip_plot_svg(records):
    """One dot per call, seconds on the x axis, one row per method (n is small, so show every point)."""
    longest = max(r["seconds"] for r in records)
    dots = []
    for row, method in enumerate(METHODS):
        y = 50 + row * 60
        dots.append(f'<text x="5" y="{y + 4}" font-size="12">{method}</text>')
        for r in (r for r in records if r["method"] == method):
            x = 90 + r["seconds"] / longest * 450
            colour = "#2a7" if r["bucket"] == "valid" else "#c33"
            dots.append(
                f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{colour}" fill-opacity="0.7">'
                f"<title>prompt {r['prompt_id']}: {r['bucket']}, {r['seconds']} s</title></circle>"
            )
    axis = f'<text x="90" y="150" font-size="12">0 s</text><text x="470" y="150" font-size="12">{longest} s</text>'
    note = '<text x="90" y="170" font-size="11">green = valid, red = malformed or truncated</text>'
    return f'<svg width="600" height="180" role="img">{"".join(dots)}{axis}{note}</svg>'


def cell(record):
    label = f"{record['method']}: {record['bucket']}, {record['seconds']} s"
    if record["bucket"] == "valid":
        body = f'<iframe srcdoc="{escape(record["html"], quote=True)}" width="100%" height="320"></iframe>'
    else:
        body = f"<p>{escape(record['reason'])}</p><pre>{escape(record['text'])}</pre>"
    return f"<td valign='top' width='50%'><b>{escape(label)}</b>{body}</td>"


def write_gallery(data, out_dir):
    """index.html: the seconds plot, then for each prompt Method A and Method B side by side.
    Valid pages are rendered; failures are shown as raw text so you can see how they failed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = data["records"]
    rows = []
    for prompt_id in sorted({r["prompt_id"] for r in records}):
        pair = {r["method"]: r for r in records if r["prompt_id"] == prompt_id}
        prompt = escape(pair[METHODS[0]]["prompt"])
        rows.append(
            f"<h3>Prompt {prompt_id}: {prompt}</h3><table width='100%'><tr>{''.join(cell(pair[m]) for m in METHODS)}</tr></table>"
        )
    page = f"<html><body style='font-family:sans-serif'><h2>Seconds per call</h2>{strip_plot_svg(records)}{''.join(rows)}</body></html>"
    (out_dir / "index.html").write_text(page)


def main(path):
    data = json.loads(Path(path).read_text())
    summary = summarize(data["records"])
    print(format_table(summary))
    name = Path(path).stem.removeprefix("benchmark_")
    Path(path).with_name(f"summary_{name}.json").write_text(
        json.dumps(summary, indent=2)
    )
    write_gallery(data, Path("results/pages") / name)
    print(f"gallery: results/pages/{name}/index.html")


if __name__ == "__main__":
    main(sys.argv[1])
