import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, sources, teacher


def train_data(args):
    out = Path(args.out)
    try:
        teacher.ensure_output_free(out, args.force)
    except FileExistsError as e:
        sys.exit(str(e))
    rows = sources.load_all()
    sampled = data.stratified_sample(rows, args.per_category, args.seed)
    with tqdm(total=len(sampled), desc="chains") as bar:

        def tick():
            stats = teacher.timing_summary(teacher.call_log())
            calls = sum(s["calls"] for s in stats.values())
            slowest = max((s["max_seconds"] for s in stats.values()), default=0)
            bar.set_postfix(api_calls=calls, slowest_s=f"{slowest:.0f}")
            bar.update()

        try:
            summary = teacher.build_training_data(
                sampled, out, force=args.force, progress=tick
            )
        except teacher.TeacherAPIError as e:
            sys.exit(
                f"\nStopped by an API error ({e.kind}). Finished rows are saved; "
                f"rerun the same command to resume.\n{str(e)[:400]}"
            )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"wrote {summary['written']} rows to {out}; summary in {summary_path}")
    for model, stats in summary["timing"].items():
        print(
            f"{model}: {stats['calls']} calls, mean {stats['mean_seconds']:.1f}s, "
            f"max {stats['max_seconds']:.1f}s, outcomes {stats['outcomes']}"
        )
    per_row = summary["row_seconds"]
    if per_row["n"]:
        print(
            f"rows generated: {per_row['n']}, mean {per_row['mean']:.1f}s, "
            f"max {per_row['max']:.1f}s"
        )


def sample(args):
    try:
        rows = data.read_jsonl(args.train)
    except FileNotFoundError:
        sys.exit(f"{args.train} not found; run train-data first")
    picked = data.stratified_sample(rows, args.per_category, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(picked, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(picked)} rows to {out}")


def main():
    parser = argparse.ArgumentParser(description="Training-data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-data", help="build data/train.jsonl")
    train.add_argument("--per-category", type=int, required=True)
    train.add_argument("--seed", type=int, default=config.SEED)
    train.add_argument("--out", default=str(config.TRAIN_PATH))
    train.add_argument("--summary", default=str(config.TRAIN_SUMMARY_PATH))
    train.add_argument("--force", action="store_true")
    train.set_defaults(run=train_data)

    review = commands.add_parser("sample", help="sample train rows to read by hand")
    review.add_argument("--train", default=str(config.TRAIN_PATH))
    review.add_argument("--per-category", type=int, default=2)
    review.add_argument("--seed", type=int, default=config.SEED)
    review.add_argument("--out", default=str(config.REVIEW_SAMPLE_PATH))
    review.set_defaults(run=sample)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
