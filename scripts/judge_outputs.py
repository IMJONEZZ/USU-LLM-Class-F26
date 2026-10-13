import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, judge
from src import eval as evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="have Gemini judge saved model outputs"
    )
    parser.add_argument("--split", required=True, choices=["eval", "test"])
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows_by_id = {row["id"]: row for row in evaluation.load_split_rows(args.split)}
    graded = data.read_jsonl(args.outputs)
    rows = [rows_by_id[g["id"]] for g in graded]

    verdicts = judge.judge_all(rows, graded)
    metrics = judge.judged_metrics(graded, verdicts)
    saved = {
        "metrics": metrics,
        "rows": [
            {"id": g["id"], "final_answer": g["final_answer"], "judge": v}
            for g, v in zip(graded, verdicts, strict=True)
        ],
    }
    evaluation.save_result(args.out, saved)

    print(f"{'category':17} {'exact':5} {'judge':5} | key | final answer")
    for row, g, verdict in zip(rows, graded, verdicts, strict=True):
        print(
            f"{row['category']:17} {g['correct']!s:5} {verdict!s:5} | "
            f"{row['answer']!r} | {g['final_answer']!r}"
        )
    print(f"\njudged accuracy {metrics['accuracy']} on {metrics['n']} rows")
    for category, cell in metrics["per_category"].items():
        print(f"  {category}: {cell['correct']}/{cell['n']} ({cell['accuracy']})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
