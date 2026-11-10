"""Generate a reproducible arithmetic dataset for prompt optimization."""

import argparse
import json
import random
from pathlib import Path


def generate_arithmetic_dataset(count=50, seed=42):
    """Return arithmetic question/answer rows generated from a fixed seed."""
    if count <= 0:
        raise ValueError("count must be positive")

    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        number1 = rng.randint(10_000, 99_999)
        number2 = rng.randint(10_000, 99_999)
        number3 = rng.randint(1_000, 9_999)
        answer = number1 + number2 - number3
        rows.append(
            {
                "question": f"What is {number1} + {number2} - {number3}?",
                "answer": str(answer),
            }
        )
    return rows


def save_arithmetic_dataset(rows, output_path):  # pragma: no cover
    """Write question/answer rows to a UTF-8 JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/arithmetic.jsonl"))
    args = parser.parse_args()

    rows = generate_arithmetic_dataset(args.count, args.seed)
    save_arithmetic_dataset(rows, args.output)
    print(f"Wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
