"""Reconstruct four-word Star Wars spans with BERT; evaluate frozen predictions.

Run ``uv run python -m src.evaluator --help``. Model/metric downloads happen only
when requested, never on import. Unit tests can supply local tokenizers and models.
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

MODEL_ID = "google-bert/bert-base-uncased"
BERTSCORE_MODEL = "roberta-base"
# Unicode letters/digits, with internal straight/curly apostrophes or hyphens.
WORD = re.compile(r"[^\W_]+(?:['’\-‐‑][^\W_]+)*", re.UNICODE)
SPLITS = ("train", "dev", "test")


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Dialogue:
    example_id: str
    character: str
    text: str
    group: str | None = None


@dataclass(frozen=True)
class MaskedExample:
    example_id: str
    text: str
    input_ids: tuple[int, ...]
    mask_positions: tuple[int, ...]
    reference_ids: tuple[int, ...]
    word_start: int
    char_start: int
    char_end: int


def load_dialogues(path):
    """Read Character/Line records; retain optional scene/conversation grouping.

    Content-based IDs survive row reordering. Duplicate rows have occurrence
    suffixes, so every source record still has its own persisted membership.
    """
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Corpus must be a nonempty JSON array")
    for row in records:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) for key in ("Character", "Line")
        ):
            raise ValueError("Every record needs string Character and Line fields")
    group_key = next(
        (
            key
            for key in ("Conversation", "conversation", "Scene", "scene")
            if any(key in row for row in records)
        ),
        None,
    )
    if group_key and any(
        type(row.get(group_key)) not in (str, int) or row[group_key] == ""
        for row in records
    ):
        raise ValueError(f"Every record must have a nonempty {group_key} identifier")
    occurrences = Counter()
    dialogues = []
    for row in records:
        # Episode qualifies scene IDs that might restart in each film.
        group = (
            json.dumps([row.get("Episode", row.get("episode")), row[group_key]])
            if group_key
            else None
        )
        identity = _digest([row["Character"], row["Line"], group])
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        dialogues.append(
            Dialogue(f"{identity}:{occurrence}", row["Character"], row["Line"], group)
        )
    return dialogues


def _validate_dialogues(dialogues):
    if not dialogues:
        raise ValueError("No dialogue examples supplied")
    ids = [row.example_id for row in dialogues]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Example IDs must be nonempty and unique")
    if any(not isinstance(row.text, str) for row in dialogues):
        raise ValueError("Dialogue text must be a string")


def load_or_create_splits(dialogues, path, seed=0):
    """Persist a deterministic approximately 80/10/10 group or line split.

    A manifest is authoritative on reuse. Reject corpus drift, missing IDs,
    invalid labels, and group leakage instead of silently replacing it.
    """
    _validate_dialogues(dialogues)
    _integer("seed", seed)
    grouped = [row.group is not None for row in dialogues]
    if any(grouped) and not all(grouped):
        raise ValueError("Grouping must be supplied for all records or none")
    units = {
        row.example_id: row.group if all(grouped) else row.example_id
        for row in dialogues
    }
    fingerprint = _digest(
        sorted((asdict(row) for row in dialogues), key=lambda row: row["example_id"])
    )
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise ValueError("Unsupported split manifest")
        if manifest.get("dataset_sha256") != fingerprint:
            raise ValueError("Split manifest does not match this dataset")
        _integer("manifest seed", manifest.get("seed"))
        expected_unit = "group" if all(grouped) else "line"
        if manifest.get("split_unit") != expected_unit:
            raise ValueError("Split manifest grouping does not match this dataset")
        assignments = manifest.get("assignments")
        if not isinstance(assignments, dict) or set(assignments) != set(units):
            raise ValueError("Split manifest must assign every example exactly once")
        if any(value not in SPLITS for value in assignments.values()):
            raise ValueError("Invalid split label in manifest")
        groups = {}
        for example_id, group in units.items():
            label = assignments[example_id]
            if groups.setdefault(group, label) != label:
                raise ValueError("A dialogue group crosses split boundaries")
        if set(assignments.values()) != set(SPLITS):
            raise ValueError("Train, dev, and test partitions must all be nonempty")
        return manifest
    ordered = sorted(
        set(units.values()), key=lambda unit: (_digest([seed, unit]), unit)
    )
    if len(ordered) < 3:
        raise ValueError("Need at least three dialogue groups/lines for three splits")
    n_train = min(len(ordered) - 2, max(1, len(ordered) * 8 // 10))
    n_dev = max(1, len(ordered) // 10)
    labels = {
        unit: ("train" if i < n_train else "dev" if i < n_train + n_dev else "test")
        for i, unit in enumerate(ordered)
    }
    manifest = {
        "schema_version": 1,
        "dataset_sha256": fingerprint,
        "seed": seed,
        "split_unit": "group" if all(grouped) else "line",
        "assignments": {key: labels[value] for key, value in sorted(units.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally overwriting an established split.
    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    return manifest


def prepare_examples(dialogues, tokenizer, seed=0, max_length=256):
    """Mask four lexical words and internal punctuation, using character offsets.

    Inputs contain only Line text, so speaker labels cannot become targets.
    Each candidate needs two visible lexical words on both sides. Candidates
    touching a special/unknown token are excluded to keep references non-special.
    """
    _validate_dialogues(dialogues)
    _integer("seed", seed)
    _integer("max_length", max_length, 1)
    if not tokenizer.is_fast or tokenizer.mask_token_id is None:
        raise ValueError("A fast tokenizer with a mask token is required")
    special = set(tokenizer.all_special_ids)
    examples, excluded = [], []
    for row in sorted(dialogues, key=lambda row: row.example_id):
        encoded = tokenizer(
            row.text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=False,
        )
        ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        words = list(WORD.finditer(row.text))
        candidates = []
        reason = None
        if len(ids) > max_length:
            reason = "overlength"
        elif len(words) < 8:
            reason = "insufficient_words"
        else:
            for start in range(2, len(words) - 5):
                left, right = words[start].start(), words[start + 3].end()
                positions = tuple(
                    i for i, (a, b) in enumerate(offsets) if a < right and b > left
                )
                if positions and all(
                    ids[i] not in special
                    and not encoded["special_tokens_mask"][i]
                    and offsets[i][0] >= left
                    and offsets[i][1] <= right
                    for i in positions
                ):
                    candidates.append((start, left, right, positions))
            if not candidates:
                reason = "no_maskable_span"
        if reason:
            excluded.append({"example_id": row.example_id, "reason": reason})
            continue
        selection = int(_digest([seed, row.example_id]), 16) % len(candidates)
        start, left, right, positions = candidates[selection]
        references = tuple(ids[i] for i in positions)
        masked = list(ids)
        for i in positions:
            masked[i] = tokenizer.mask_token_id
        examples.append(
            MaskedExample(
                row.example_id,
                row.text,
                tuple(masked),
                positions,
                references,
                start,
                left,
                right,
            )
        )
    counts = dict(Counter(item["reason"] for item in excluded))
    return examples, {
        "source_count": len(dialogues),
        "eligible_count": len(examples),
        "excluded_count": len(excluded),
        "excluded_by_reason": counts,
        "excluded_examples": excluded,
    }


def predict_spans(examples, model, tokenizer, batch_size=8, device="cpu"):
    """One simultaneous, unfiltered argmax per masked position; right padding."""
    _integer("batch_size", batch_size, 1)
    if not examples:
        raise ValueError("No eligible examples to predict")
    if tokenizer.padding_side != "right" or tokenizer.pad_token_id is None:
        raise ValueError("BERT inference requires right padding and a pad token")
    for example in examples:
        positions = example.mask_positions
        for position in positions:
            _integer("mask position", position)
        for token in (*example.input_ids, *example.reference_ids):
            _integer("token ID", token)
        if (
            not positions
            or len(positions) != len(example.reference_ids)
            or tuple(sorted(set(positions))) != positions
            or any(i < 0 or i >= len(example.input_ids) for i in positions)
            or any(example.input_ids[i] != tokenizer.mask_token_id for i in positions)
        ):
            raise ValueError("Invalid masked example")
    model.to(device)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            inputs = tokenizer.pad(
                [
                    {
                        "input_ids": list(row.input_ids),
                        "attention_mask": [1] * len(row.input_ids),
                    }
                    for row in batch
                ],
                padding=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            logits = model(**inputs).logits
            if logits.ndim != 3 or logits.shape[:2] != inputs["input_ids"].shape:
                raise ValueError("Model logits do not align with input positions")
            for row, example in zip(logits, batch, strict=True):
                selected = row[list(example.mask_positions)]
                if not torch.isfinite(selected).all():
                    raise ValueError("Model returned non-finite masked-token logits")
                predictions.append(selected.argmax(dim=-1).cpu().tolist())
    return predictions


def strict_metrics(predictions, references):
    """Micro token accuracy and fraction of exactly reconstructed token spans."""
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions/references must be nonempty with matching counts")
    correct = total = exact = 0
    for predicted, reference in zip(predictions, references, strict=True):
        if not reference or len(predicted) != len(reference):
            raise ValueError(
                "Each predicted span must match its nonempty reference length"
            )
        for token in (*predicted, *reference):
            _integer("token ID", token)
        matches = sum(a == b for a, b in zip(predicted, reference, strict=True))
        correct += matches
        total += len(reference)
        exact += matches == len(reference)
    return {
        "token_accuracy": correct / total,
        "exact_span_match": exact / len(references),
        "correct_tokens": correct,
        "masked_tokens": total,
        "exact_spans": exact,
        "example_count": len(references),
    }


def baseline_token(dialogues, tokenizer):
    """Most frequent non-special training token; ties favor the smaller ID."""
    _validate_dialogues(dialogues)
    special = set(tokenizer.all_special_ids)
    counts = Counter(
        token
        for row in dialogues
        for token in tokenizer(row.text, add_special_tokens=False)["input_ids"]
        if token not in special
    )
    if not counts:
        raise ValueError(
            "Training partition has no non-special tokens for the baseline"
        )
    return min(counts, key=lambda token: (-counts[token], token))


def text_metrics(
    predictions, references, batch_size=8, device="cpu", metric_loader=None
):
    """Hugging Face metrics on decoded spans, never unchanged surrounding text."""
    _integer("batch_size", batch_size, 1)
    if not predictions or len(predictions) != len(references):
        raise ValueError(
            "Text predictions/references must have matching nonempty counts"
        )
    if any(
        not isinstance(text, str) or not text.strip()
        for text in [*predictions, *references]
    ):
        raise ValueError("Text metrics require nonempty decoded strings")
    if metric_loader is None:
        from evaluate import load

        metric_loader = load
    rouge = metric_loader("rouge").compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rougeL"],
        use_stemmer=False,
        use_aggregator=False,
    )
    # Hugging Face can return None on non-main distributed processes.
    # This evaluator expects concrete scores for its single-process report.
    if rouge is None:
        raise RuntimeError("ROUGE returned no result; run metrics on the main process")
    # Mean per-example F1 avoids ROUGE's stochastic bootstrap aggregation.
    rouge = {name: sum(values) / len(values) for name, values in rouge.items()}
    bleu = metric_loader("bleu").compute(
        predictions=predictions,
        references=[[text] for text in references],
        max_order=2,
        smooth=True,
    )
    if bleu is None:
        raise RuntimeError("BLEU returned no result; run metrics on the main process")
    bertscore = metric_loader("bertscore").compute(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        batch_size=batch_size,
        device=device,
        idf=False,
        rescale_with_baseline=False,
    )
    if bertscore is None:
        raise RuntimeError(
            "BERTScore returned no result; run metrics on the main process"
        )
    return {
        "rouge": rouge,
        "bleu": bleu,
        "bertscore": {
            key: sum(bertscore[key]) / len(bertscore[key])
            for key in ("precision", "recall", "f1")
        },
        "bertscore_hash": bertscore["hashcode"],
    }


def _decode(tokenizer, ids):
    # Never drop a predicted special token: it is part of the model's answer.
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def evaluate_examples(
    examples,
    training,
    model,
    tokenizer,
    batch_size=8,
    device="cpu",
    include_text_metrics=True,
    metric_loader=None,
):
    """Run inference once and return metrics plus all auditable per-example data."""
    token = baseline_token(training, tokenizer)
    predictions = predict_spans(examples, model, tokenizer, batch_size, device)
    references = [list(row.reference_ids) for row in examples]
    baseline = [[token] * len(reference) for reference in references]
    rows = []
    for example, prediction in zip(examples, predictions, strict=True):
        rows.append(
            {
                **asdict(example),
                "prediction_ids": prediction,
                "prediction": _decode(tokenizer, prediction),
                "reference": _decode(tokenizer, example.reference_ids),
                "masked_text": example.text[: example.char_start]
                + "[MISSING SPAN]"
                + example.text[example.char_end :],
            }
        )
    result = {
        "bert": strict_metrics(predictions, references),
        "baseline": {
            "token_id": token,
            "token": _decode(tokenizer, [token]),
            **strict_metrics(baseline, references),
        },
        "examples": rows,
    }
    if include_text_metrics:
        result["text_metrics"] = text_metrics(
            [row["prediction"] for row in rows],
            [row["reference"] for row in rows],
            batch_size,
            device,
            metric_loader,
        )
    return result


def _versions():
    versions = {"python": sys.version.split()[0]}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "evaluate",
        "datasets",
        "bert-score",
        "rouge-score",
        "numpy",
    ):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", default="SW_EpisodeIV_VI.json")
    parser.add_argument(
        "--splits", type=Path, default=Path("data/evaluation/splits.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/results.json")
    )
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--sample-size", type=int, help="Fixed dev subset only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--device", default="cpu", help="cpu or a PyTorch device such as cuda"
    )
    parser.add_argument(
        "--revision", default="main", help="BERT Hub revision or commit"
    )
    parser.add_argument(
        "--strict-only", action="store_true", help="Skip text metrics for debugging"
    )
    args = parser.parse_args(argv)
    for name in ("batch_size", "max_length"):
        _integer(name, getattr(args, name), 1)
    _integer("seed", args.seed)
    if args.sample_size is not None:
        _integer("sample_size", args.sample_size, 1)
        if args.split != "dev":
            parser.error(
                "--sample-size is only allowed with --split dev; test uses all eligible examples"
            )
    if args.output.resolve() in (args.splits.resolve(), Path(args.corpus).resolve()):
        parser.error("Output must not overwrite the corpus or split manifest")
    dialogues = load_dialogues(args.corpus)
    manifest = load_or_create_splits(dialogues, args.splits, args.seed)
    assignments = manifest["assignments"]
    training = [row for row in dialogues if assignments[row.example_id] == "train"]
    selected = [row for row in dialogues if assignments[row.example_id] == args.split]
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=args.revision, use_fast=True
    )
    examples, preparation = prepare_examples(
        selected, tokenizer, args.seed, args.max_length
    )
    if not examples:
        raise ValueError(f"No eligible examples: {preparation['excluded_by_reason']}")
    if args.sample_size is not None:
        examples = examples[: args.sample_size]
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=args.revision)
    if args.max_length > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds BERT's position embedding capacity")
    print(
        f"Evaluating {len(examples)} {args.split} examples on {args.device}", flush=True
    )
    result = evaluate_examples(
        examples,
        training,
        model,
        tokenizer,
        args.batch_size,
        args.device,
        not args.strict_only,
    )
    result["metadata"] = {
        "model": MODEL_ID,
        "tokenizer": MODEL_ID,
        "requested_revision": args.revision,
        "resolved_revision": model.config._commit_hash,
        "dataset_sha256": manifest["dataset_sha256"],
        "split_manifest": str(args.splits),
        "split_seed": manifest["seed"],
        "split_unit": manifest["split_unit"],
        "split": args.split,
        "partition_counts": dict(Counter(assignments.values())),
        "seed": args.seed,
        "sample_count": len(examples),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "preparation": preparation,
        "masking": {
            "words": 4,
            "context_words_each_side": 2,
            "word_regex": WORD.pattern,
            "selection": "sha256([seed, example_id]) modulo valid span count",
            "internal_punctuation": "masked",
            "reconstruction": "simultaneous argmax",
        },
        "metric_settings": {
            "text_metrics_enabled": not args.strict_only,
            "rouge_types": ["rouge1", "rougeL"],
            "rouge_aggregation": "mean F1",
            "rouge_use_stemmer": False,
            "bleu_max_order": 2,
            "bleu_smooth": True,
            "bertscore_model": BERTSCORE_MODEL,
            "bertscore_idf": False,
            "bertscore_rescale_with_baseline": False,
        },
        "versions": _versions(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("examples", "metadata")
            },
            indent=2,
        )
    )
    print(f"Saved report to {args.output}")
    return result


if __name__ == "__main__":
    main()
