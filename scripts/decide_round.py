import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src import eval as evaluation
from src import train as training


def judged_accuracy(results_dir, round_n):
    path = Path(results_dir) / f"round{round_n}_judged.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {
        category: cell["accuracy"]
        for category, cell in saved["metrics"]["per_category"].items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="decide the next round from judged scores"
    )
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--gap", type=float, required=True)
    parser.add_argument("--drop", type=float)
    parser.add_argument("--results-dir", default=str(config.RESULTS_DIR))
    args = parser.parse_args(argv)

    accuracy = judged_accuracy(args.results_dir, args.round)
    weak = evaluation.find_weak_categories(accuracy, args.gap)
    decision = training.round_decision(weak)
    print(f"round {args.round} judged accuracy: {accuracy}")
    print(f"decision: {decision}; weak categories: {weak}")

    regressed = None
    if args.drop is not None and args.round > 0:
        before = judged_accuracy(args.results_dir, args.round - 1)
        regressed = evaluation.find_regressions(before, accuracy, args.drop)
        print(f"regressions against round {args.round - 1}: {regressed}")
    return {"weak": weak, "decision": decision, "regressed": regressed}


if __name__ == "__main__":
    main()
