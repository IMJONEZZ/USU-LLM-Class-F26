import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, data, sources, teacher, train


def train_data(args):
    out = Path(args.out)
    try:
        test_rows = data.load_test_set(args.test_set)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(f"{e}\nThe test set must be valid before building train data.")
    rows, leaked = data.drop_leaked(sources.load_all(), test_rows)
    print(f"dropped {len(leaked)} public rows that duplicate a test question")
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
                sampled, out, args.store, progress=tick
            )
        except teacher.TeacherAPIError as e:
            sys.exit(
                f"\nStopped by an API error ({e.kind}). Every finished chain is "
                f"saved in {args.store}; rerun the same command to continue.\n"
                f"{str(e)[:400]}"
            )
        except ValueError as e:
            sys.exit(str(e))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"wrote {summary['written']} rows to {out}: {summary['generated']} newly "
        f"generated, {summary['reused']} reused from {args.store}, "
        f"{len(summary['failed'])} failed (never retried); summary in {summary_path}"
    )
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


def test_data(args):
    try:
        rows = data.load_test_set(args.test_set)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    data.record_test_set_hash(args.test_set, args.hash)
    counts = dict(Counter(row["category"] for row in rows))
    print(f"test set ok: {len(rows)} rows {counts}; hash recorded in {args.hash}")


def token_audit(args, train_rows, eval_rows, test_rows):
    tokenizer = data.load_tokenizer(args.tokenizer)
    sets = {"train": train_rows, "eval": eval_rows, "test": test_rows}
    lengths = {name: data.token_lengths(rows, tokenizer) for name, rows in sets.items()}
    audit = {"tokenizer": args.tokenizer, "max_seq_length": args.max_seq_length}
    for name, items in lengths.items():
        audit[name] = data.length_summary(items) if items else {"n": 0}
        if items:
            s = audit[name]
            print(
                f"{name}: n={s['n']}, total tokens median {s['total']['p50']:.0f}, "
                f"p90 {s['total']['p90']:.0f}, max {s['total']['max']} "
                f"(prompt max {s['prompt']['max']}, response max {s['response']['max']})"
            )
    longest = max(item["total"] for items in lengths.values() for item in items)
    recommended = data.recommended_max_seq_length(longest)
    parts = train.estimate_training_memory_gib(recommended, args.batch_size)
    try:
        gpu = train.choose_gpu(parts["total"])
    except ValueError:
        gpu = None
    audit["recommended_max_seq_length"] = recommended
    audit["memory_estimate_gib"] = {k: round(v, 2) for k, v in parts.items()}
    audit["cheapest_gpu"] = gpu
    chosen = args.gpu
    capacity = config.GPU_OPTIONS[chosen]["vram_gib"] * config.GPU_USABLE_FRACTION
    fits = parts["total"] <= capacity
    audit["gpu_type"] = chosen
    audit["fits_chosen_gpu"] = fits
    path = Path(args.token_audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=1), encoding="utf-8")
    print(
        f"longest example {longest} tokens; recommended MAX_SEQ_LENGTH {recommended} "
        f"(now {args.max_seq_length})"
    )
    print(
        f"estimated training memory at {recommended} tokens, batch "
        f"{args.batch_size}: {parts['total']:.1f} GiB (a rough estimate; the "
        f"two-step probe in M8 measures it)"
    )
    if gpu is None:
        print(f"no GPU in the table fits {parts['total']:.1f} GiB; audit in {path}")
    else:
        hourly = config.GPU_OPTIONS[gpu]["usd_per_second"] * 3600
        print(f"cheapest GPU that fits: {gpu} at ${hourly:.2f}/hour; audit in {path}")
    chosen_hourly = config.GPU_OPTIONS[chosen]["usd_per_second"] * 3600
    print(
        f"chosen GPU: {chosen} at ${chosen_hourly:.2f}/hour "
        f"({'fits' if fits else 'DOES NOT FIT'}: {parts['total']:.1f} GiB needed, "
        f"{capacity:.1f} GiB usable)"
    )
    if not fits:
        sys.exit(
            f"the estimate does not fit {chosen}; pick a larger GPU or a smaller batch"
        )
    for name, items in lengths.items():
        if items:
            try:
                data.check_token_limit(items, args.max_seq_length)
            except ValueError as e:
                sys.exit(f"{name}: {e}")


def validate(args):
    try:
        test_rows = data.load_test_set(args.test_set)
        status = data.check_test_set_hash(args.test_set, args.hash)
        try:
            train_rows = data.read_jsonl(args.train)
        except FileNotFoundError:
            sys.exit(f"{args.train} not found; run train-data first")
        eval_path = Path(args.eval)
        eval_rows = data.read_jsonl(eval_path) if eval_path.exists() else []
        near = data.check_leakage(test_rows, train_rows, eval_rows)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    print(
        f"validate ok: {len(test_rows)} test rows, {len(train_rows)} train rows, "
        f"{len(eval_rows)} eval rows, test-set hash {status}, "
        f"{len(near)} near-duplicate warnings"
    )
    if args.tokens:
        token_audit(args, train_rows, eval_rows, test_rows)


def eval_data(args):
    out = Path(args.out)
    try:
        test_rows = data.load_test_set(args.test_set)
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))
    try:
        with tqdm(desc="judge calls") as bar:
            summary = teacher.build_eval_data(
                test_rows,
                out,
                args.review,
                args.store,
                per_question=args.per_question,
                progress=bar.update,
            )
    except teacher.TeacherAPIError as e:
        sys.exit(
            f"\nStopped by an API error ({e.kind}). Everything already generated "
            f"or judged is saved in {args.store}; rerun once it is fixed.\n"
            f"{str(e)[:400]}"
        )
    except (RuntimeError, ValueError) as e:
        sys.exit(str(e))
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        f"eval variants: {summary['kept']} kept, {summary['rejected']} rejected "
        f"{summary['rejected_by_reason']}; spent {summary['paraphrase_calls']} "
        f"paraphrase and {summary['judge_calls']} judge calls (the rest came from "
        f"{args.store}); summary in {summary_path}"
    )
    question_of = {row["id"]: row["question"] for row in test_rows}
    records = data.read_jsonl(out) + data.read_jsonl(args.review)
    for record in data.pick_random(records, args.show, args.seed):
        print(f"[{record['verdict']['reason']}] {record['source_id']}")
        print(f"    was: {question_of[record['source_id']]}")
        print(f"    now: {record['question']}")


def main():
    parser = argparse.ArgumentParser(description="Training-data pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train-data", help="build data/train.jsonl")
    train.add_argument("--per-category", type=int, required=True)
    train.add_argument("--seed", type=int, default=config.SEED)
    train.add_argument("--out", default=str(config.TRAIN_PATH))
    train.add_argument("--summary", default=str(config.TRAIN_SUMMARY_PATH))
    train.add_argument("--test-set", default=str(config.TEST_SET_PATH))
    train.add_argument("--store", default=str(config.CHAIN_STORE_PATH))
    train.set_defaults(run=train_data)

    review = commands.add_parser("sample", help="sample train rows to read by hand")
    review.add_argument("--train", default=str(config.TRAIN_PATH))
    review.add_argument("--per-category", type=int, default=2)
    review.add_argument("--seed", type=int, default=config.SEED)
    review.add_argument("--out", default=str(config.REVIEW_SAMPLE_PATH))
    review.set_defaults(run=sample)

    check_test = commands.add_parser(
        "test-data", help="check the test set and record its hash"
    )
    check_test.add_argument("--test-set", default=str(config.TEST_SET_PATH))
    check_test.add_argument("--hash", default=str(config.TEST_SET_HASH_PATH))
    check_test.set_defaults(run=test_data)

    check_all = commands.add_parser(
        "validate", help="check the test set and look for leakage into train/eval"
    )
    check_all.add_argument("--test-set", default=str(config.TEST_SET_PATH))
    check_all.add_argument("--train", default=str(config.TRAIN_PATH))
    check_all.add_argument("--eval", default=str(config.EVAL_PATH))
    check_all.add_argument("--hash", default=str(config.TEST_SET_HASH_PATH))
    check_all.add_argument(
        "--tokens",
        action="store_true",
        help="also audit token lengths (needs the tokenizer)",
    )
    check_all.add_argument("--tokenizer", default=config.MODEL_NAME)
    check_all.add_argument("--max-seq-length", type=int, default=config.MAX_SEQ_LENGTH)
    check_all.add_argument("--token-audit", default=str(config.TOKEN_AUDIT_PATH))
    check_all.add_argument("--batch-size", type=int, default=1)
    check_all.add_argument(
        "--gpu", default=config.GPU_TYPE, choices=sorted(config.GPU_OPTIONS)
    )
    check_all.set_defaults(run=validate)

    variants = commands.add_parser(
        "eval-data", help="build data/eval.jsonl from reworded test questions"
    )
    variants.add_argument("--test-set", default=str(config.TEST_SET_PATH))
    variants.add_argument("--out", default=str(config.EVAL_PATH))
    variants.add_argument("--review", default=str(config.EVAL_REVIEW_PATH))
    variants.add_argument("--summary", default=str(config.EVAL_SUMMARY_PATH))
    variants.add_argument(
        "--per-question", type=int, default=config.EVAL_VARIANTS_PER_QUESTION
    )
    variants.add_argument("--seed", type=int, default=config.SEED)
    variants.add_argument("--show", type=int, default=20)
    variants.add_argument("--store", default=str(config.EVAL_STORE_PATH))
    variants.set_defaults(run=eval_data)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
