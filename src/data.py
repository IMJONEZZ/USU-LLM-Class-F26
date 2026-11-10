import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from src import config


def to_example(row):
    """Turn one IFEval row into {key, prompt, instruction_id, kwargs}, or None if it has more than one instruction."""
    if len(row["instruction_id_list"]) != 1:
        return None
    # the dataset gives every instruction all 24 argument names; only the non-None ones are real
    kwargs = {k: v for k, v in row["kwargs"][0].items() if v is not None}
    return {
        "key": row["key"],
        "prompt": row["prompt"],
        "instruction_id": row["instruction_id_list"][0],
        "kwargs": kwargs,
    }


def drop_small_types(examples, min_prompts):
    """Return (kept examples, {dropped type: prompt count}); a type with min_prompts or fewer prompts is dropped."""
    sizes = Counter(e["instruction_id"] for e in examples)
    dropped = {t: n for t, n in sizes.items() if n <= min_prompts}
    return [e for e in examples if e["instruction_id"] not in dropped], dropped


def split_examples(examples, seed):
    """Stratified train/validation/test split by instruction_id; returns {split name: [keys]}."""
    total = len(examples)
    # row counts, not fractions: sklearn rounds float sizes up or down, which could move a row between splits
    train_size = round(total * config.TRAIN_FRACTION)
    test_size = round(total * config.TEST_FRACTION)
    train, rest = train_test_split(
        examples,
        train_size=train_size,
        stratify=[e["instruction_id"] for e in examples],
        random_state=seed,
    )
    validation, test = train_test_split(
        rest,
        test_size=test_size,
        stratify=[e["instruction_id"] for e in rest],
        random_state=seed,
    )
    return {
        "train": [e["key"] for e in train],
        "validation": [e["key"] for e in validation],
        "test": [e["key"] for e in test],
    }


def split_counts(examples, splits):
    """Return {instruction_id: {split name: count}}."""
    type_of = {e["key"]: e["instruction_id"] for e in examples}
    counts = {t: dict.fromkeys(splits, 0) for t in dict.fromkeys(type_of.values())}
    for name, keys in splits.items():
        for key in keys:
            counts[type_of[key]][name] += 1
    return counts


def load_examples():  # pragma: no cover - needs the network
    from datasets import load_dataset

    rows = load_dataset("google/IFEval", split="train")
    return rows, [e for e in map(to_example, rows) if e is not None]


def main():  # pragma: no cover - needs the network
    rows, single = load_examples()
    examples, dropped = drop_small_types(single, config.MIN_TYPE_PROMPTS)
    splits = split_examples(examples, config.SEED)
    path = Path(config.SPLITS_PATH)
    if path.exists():
        same = json.loads(path.read_text()) == splits
        print(f"{path} exists; matches this run's split: {same}")
    else:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(splits, indent=2))
        print(f"wrote {path}")

    counts = split_counts(examples, splits)
    types = len(counts)
    print(
        f"rows {len(rows)} -> single-instruction {len(single)} -> kept {len(examples)} ({types} types)"
    )
    print(f"dropped types: {dropped}")
    print({name: len(keys) for name, keys in splits.items()})
    print(f"{'type':55} train  val  test")
    for t, c in sorted(counts.items(), key=lambda item: -sum(item[1].values())):
        print(f"{t:55} {c['train']:5} {c['validation']:4} {c['test']:5}")

    print("\nhand-check sample (smallest, middle, largest type):")
    ordered = sorted(counts, key=lambda t: sum(counts[t].values()))
    for t in (ordered[0], ordered[types // 2], ordered[-1]):
        for e in [x for x in examples if x["instruction_id"] == t][:2]:
            print(
                f"  {t} | key {e['key']} | kwargs {e['kwargs']} | {e['prompt'][:70]!r}"
            )


if __name__ == "__main__":  # pragma: no cover
    main()
