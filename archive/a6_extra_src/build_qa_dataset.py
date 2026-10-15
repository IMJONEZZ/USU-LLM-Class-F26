"""Assignment 6 QA data: download, leakage inspection, conversion helpers.

Phase 1:
  download BrainTeaser / FalseQA / StrategyQA / TruthfulQA
  scan questions against GRADED_QUESTIONS
  write data/assignment6/leakage_report.md

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.trainer import GRADED_QUESTIONS, normalize_text, validate_qa_record

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "assignment6"
RAW = DATA / "raw"
QA = DATA / "qa"

# Distinctive scenario cues for same-scenario / paraphrase suspicion.
SCENARIO_CUES = {
    "opera_people": {
        "stem_index": 0,
        "all_of": (("opera",), ("people",), ("hour",)),
    },
    "feathers_pound": {
        "stem_index": 1,
        "all_of": (("feather",), ("pound",)),
    },
    "christmas_tree_boxes": {
        "stem_index": 2,
        "all_of": (("tree",), ("box", "boxes"), ("living", "stair", "morning")),
    },
    "knuckle_crack": {
        "stem_index": 3,
        "all_of": (("knuckle",), ("crack",)),
    },
    "basement_shark": {
        "stem_index": 4,
        "all_of": (("shark",), ("basement", "pool", "upstairs")),
    },
    "woodchuck": {
        "stem_index": 5,
        "all_of": (("woodchuck", "wood chuck", "wood-chuck"),),
    },
    "us_president": {
        "stem_index": 6,
        "all_of": (("president",), ("united states", "u.s.", "usa", "american")),
    },
    "talos": {
        "stem_index": 7,
        "all_of": (("talos",),),
    },
    "parallel_letters": {
        "stem_index": 8,
        "all_of": (("parallel",),),
    },
    "sphinx_riddle": {
        "stem_index": 9,
        "all_of": (("sphinx",),),
    },
}


def _get(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "usu-llm-class-f26-a6/1.0"})
    with urlopen(request, timeout=180) as response:
        return response.read()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def download_falseqa() -> Path:
    out = RAW / "falseqa"
    out.mkdir(parents=True, exist_ok=True)
    candidates = [
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/train.csv",
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/dev.csv",
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/dataset/test.csv",
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/data/train.csv",
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/data/dev.csv",
        "https://raw.githubusercontent.com/thunlp/FalseQA/main/data/test.csv",
    ]
    saved = []
    for url in candidates:
        name = url.rsplit("/", 1)[-1]
        target = out / name
        if target.exists() and target.stat().st_size > 0:
            saved.append(target)
            continue
        try:
            payload = _get(url)
        except OSError, URLError, HTTPError, TimeoutError:
            continue
        if payload.startswith(b"404") or b"Not Found" in payload[:120]:
            continue
        if b"<html" in payload[:200].lower():
            continue
        _write_bytes(target, payload)
        saved.append(target)
    if not any(p.stat().st_size > 0 for p in out.glob("*.csv")):
        raise RuntimeError("Could not download FalseQA CSVs from thunlp/FalseQA")
    return out


def download_brainteaser() -> Path:
    out = RAW / "brainteaser"
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "BTDATA.zip"
    urls = [
        "https://github.com/1171-jpg/BrainTeaser/raw/main/data/BTDATA.zip",
        "https://github.com/1171-jpg/BrainTeaser/raw/master/data/BTDATA.zip",
        "https://github.com/1171-jpg/BrainTeaser/raw/main/BTDATA.zip",
    ]
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        last_error = None
        for url in urls:
            try:
                payload = _get(url)
                if payload.startswith(b"PK") or len(payload) > 1000:
                    _write_bytes(zip_path, payload)
                    break
            except Exception as error:  # noqa: BLE001
                last_error = error
        else:
            raise RuntimeError(
                f"Could not download BrainTeaser BTDATA.zip: {last_error}"
            )
    extract_dir = out / "extracted"
    marker = extract_dir / ".extracted_ok"
    if not marker.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        # BTDATA.zip uses WinZip AES (compress_type=99); stdlib zipfile cannot open it.
        try:
            import pyzipper

            with pyzipper.AESZipFile(zip_path) as archive:
                archive.pwd = b"brainteaser"
                archive.extractall(extract_dir)
        except ImportError as error:
            raise RuntimeError(
                "BrainTeaser zip is AES-encrypted; install pyzipper "
                "(uv run --with pyzipper ...)"
            ) from error
        marker.write_text("ok\n", encoding="utf-8")
    return extract_dir


def download_strategyqa() -> Path:
    out = RAW / "strategyqa"
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    dataset = load_dataset("ChilleD/StrategyQA")
    for split, table in dataset.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in table:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return out


def download_truthfulqa() -> Path:
    out = RAW / "truthfulqa"
    out.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset

    dataset = load_dataset("truthfulqa/truthful_qa", "generation")
    for split, table in dataset.items():
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in table:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return out


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_questions_falseqa(root: Path) -> list[dict]:
    items = []
    for csv_path in sorted(root.glob("*.csv")):
        text = csv_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for index, row in enumerate(reader):
            question = (row.get("question") or row.get("Question") or "").strip()
            if not question:
                continue
            items.append(
                {
                    "dataset": "falseqa",
                    "example_id": f"falseqa:{csv_path.stem}:{index}",
                    "question": question,
                    "split": csv_path.stem,
                }
            )
    return items


def _extract_questions_strategyqa(root: Path) -> list[dict]:
    items = []
    for path in sorted(root.glob("*.jsonl")):
        for index, row in enumerate(_load_jsonl(path)):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            qid = row.get("qid") or f"{path.stem}:{index}"
            items.append(
                {
                    "dataset": "strategyqa",
                    "example_id": f"strategyqa:{qid}",
                    "question": question,
                    "split": path.stem,
                }
            )
    return items


def _extract_questions_truthfulqa(root: Path) -> list[dict]:
    items = []
    for path in sorted(root.glob("*.jsonl")):
        for index, row in enumerate(_load_jsonl(path)):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            items.append(
                {
                    "dataset": "truthfulqa",
                    "example_id": f"truthfulqa:{path.stem}:{index}",
                    "question": question,
                    "split": path.stem,
                    "category": row.get("category"),
                }
            )
    return items


def _brainteaser_rows_from_obj(obj, *, prefix: str) -> list[dict]:
    items = []
    if isinstance(obj, dict):
        question = (
            obj.get("question")
            or obj.get("Question")
            or obj.get("puzzle")
            or obj.get("text")
        )
        if isinstance(question, str) and question.strip():
            items.append(
                {
                    "dataset": "brainteaser",
                    "example_id": f"brainteaser:{prefix}",
                    "question": question.strip(),
                    "split": prefix,
                }
            )
        for key, value in obj.items():
            if key in {"question", "Question", "puzzle", "text", "answer", "label"}:
                continue
            items.extend(_brainteaser_rows_from_obj(value, prefix=f"{prefix}:{key}"))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            items.extend(_brainteaser_rows_from_obj(value, prefix=f"{prefix}:{index}"))
    elif isinstance(obj, str) and "?" in obj and len(obj) > 12:
        items.append(
            {
                "dataset": "brainteaser",
                "example_id": f"brainteaser:{prefix}",
                "question": obj.strip(),
                "split": prefix,
            }
        )
    return items


def _extract_questions_brainteaser(root: Path) -> list[dict]:
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        rel = str(path.relative_to(root))
        try:
            if suffix == ".json":
                obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                items.extend(_brainteaser_rows_from_obj(obj, prefix=rel))
            elif suffix == ".jsonl":
                for index, row in enumerate(_load_jsonl(path)):
                    question = (
                        row.get("question")
                        or row.get("Question")
                        or row.get("puzzle")
                        or ""
                    )
                    if question:
                        items.append(
                            {
                                "dataset": "brainteaser",
                                "example_id": f"brainteaser:{rel}:{index}",
                                "question": str(question).strip(),
                                "split": rel,
                            }
                        )
            elif suffix == ".npy":
                import numpy as np

                arr = np.load(path, allow_pickle=True)
                obj = arr.tolist() if hasattr(arr, "tolist") else arr
                items.extend(_brainteaser_rows_from_obj(obj, prefix=rel))
        except OSError, ValueError, TypeError, json.JSONDecodeError, ImportError:
            continue
    # Deduplicate identical questions within brainteaser.
    seen = set()
    unique = []
    for item in items:
        key = normalize_text(item["question"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def collect_all_questions() -> list[dict]:
    items: list[dict] = []
    falseqa = RAW / "falseqa"
    brain = RAW / "brainteaser" / "extracted"
    strategy = RAW / "strategyqa"
    truthful = RAW / "truthfulqa"
    if falseqa.exists():
        items.extend(_extract_questions_falseqa(falseqa))
    if brain.exists():
        items.extend(_extract_questions_brainteaser(brain))
    if strategy.exists():
        items.extend(_extract_questions_strategyqa(strategy))
    if truthful.exists():
        items.extend(_extract_questions_truthfulqa(truthful))
    return items


def _cue_hit(text: str, alternatives: tuple[str, ...]) -> bool:
    return any(alt in text for alt in alternatives)


def scenario_matches(question: str) -> list[dict]:
    text = normalize_text(question)
    hits = []
    for name, spec in SCENARIO_CUES.items():
        groups = spec["all_of"]
        if not all(_cue_hit(text, group) for group in groups):
            continue
        # Require letter-counting intent beyond just mentioning "parallel".
        if name == "parallel_letters" and not any(
            token in text
            for token in ("how many", "number of", "count", "letter", " l ", "ls")
        ):
            continue
        hits.append(
            {
                "scenario": name,
                "stem_index": spec["stem_index"] + 1,
                "graded_stem": GRADED_QUESTIONS[spec["stem_index"]],
            }
        )
    return hits


def inspect_item(item: dict, *, jaccard_threshold: float = 0.55) -> list[dict]:
    """Return exclusion findings.

    Policy (per assignment owner): only normalized **exact** matches are excluded.
    Jaccard / containment / same-scenario heuristics are not used for exclusion.
    """
    del (
        jaccard_threshold
    )  # kept for call-site compatibility; unused under exact-only policy
    question = item.get("question") or ""
    if not question.strip():
        return []
    text = normalize_text(question)
    findings = []
    for index, stem in enumerate(GRADED_QUESTIONS):
        stem_norm = normalize_text(stem)
        if text == stem_norm:
            findings.append(
                {
                    **item,
                    "match_type": "exact",
                    "stem_index": index + 1,
                    "graded_stem": stem,
                    "score": 1.0,
                    "action": "excluded",
                }
            )
    return findings


def write_leakage_report(findings: list[dict], inventory: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset: dict[str, list[dict]] = {}
    for row in findings:
        by_dataset.setdefault(row["dataset"], []).append(row)

    lines = [
        "# Assignment 6 — graded-question leakage inspection",
        "",
        "Phase 1 report only.",
        "**Exclusion policy:** normalized exact matches only (case/whitespace-insensitive).",
        "Containment, Jaccard, and same-scenario heuristics are not used for exclusion.",
        "No examples were auto-approved.",
        "",
        "## Inventory",
        "",
        "| Dataset | Questions scanned |",
        "| --- | ---: |",
    ]
    for name, count in sorted(inventory.items()):
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        f"**Total exact-match exclusions:** {len(findings)}",
        "",
        "## Findings (exact only)",
        "",
    ]
    if not findings:
        lines.append("No exact matches found.")
    else:
        for dataset, rows in sorted(by_dataset.items()):
            lines.append(f"### {dataset} ({len(rows)})")
            lines.append("")
            for row in rows:
                q = row["question"].replace("\n", " ")
                if len(q) > 220:
                    q = q[:217] + "..."
                score = row.get("score")
                score_s = f", score={score}" if score is not None else ""
                scenario = row.get("scenario")
                scenario_s = f", scenario=`{scenario}`" if scenario else ""
                lines.append(
                    f"- **{row['example_id']}** — `{row['match_type']}` "
                    f"vs graded Q{row['stem_index']}{score_s}{scenario_s}; "
                    f"**action:** {row['action']}"
                )
                lines.append(f"  - Question: {q}")
                lines.append(f"  - Graded stem: {row['graded_stem']}")
            lines.append("")

    lines += ["## Graded stems checked", ""]
    for index, stem in enumerate(GRADED_QUESTIONS, start=1):
        lines.append(f"{index}. {stem}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_download(_: argparse.Namespace) -> None:
    print("Downloading FalseQA...")
    print(" ", download_falseqa())
    print("Downloading BrainTeaser...")
    print(" ", download_brainteaser())
    print("Downloading StrategyQA...")
    print(" ", download_strategyqa())
    print("Downloading TruthfulQA...")
    print(" ", download_truthfulqa())
    print("Done. Raw files under", RAW)


def cmd_inspect(_: argparse.Namespace) -> None:
    items = collect_all_questions()
    if not items:
        raise SystemExit(
            f"No questions found under {RAW}. Run: python -m src.build_qa_dataset download"
        )
    inventory: dict[str, int] = {}
    for item in items:
        inventory[item["dataset"]] = inventory.get(item["dataset"], 0) + 1
    findings: list[dict] = []
    for item in items:
        findings.extend(inspect_item(item))
    report = DATA / "leakage_report.md"
    write_leakage_report(findings, inventory, report)
    (DATA / "leakage_findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Scanned {len(items)} questions across {len(inventory)} datasets")
    for name, count in sorted(inventory.items()):
        print(f"  {name}: {count}")
    print(f"Findings: {len(findings)}")
    print(f"Wrote {report}")
    print(f"Wrote {DATA / 'leakage_findings.json'}")
    print("STOP — review the leakage report before Phase 2 conversion.")


def _excluded_ids() -> set[str]:
    path = DATA / "leakage_findings.json"
    if not path.exists():
        return set()
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        row["example_id"]
        for row in rows
        if row.get("action") == "excluded" and row.get("example_id")
    }


def _qa_record(
    *,
    example_id: str,
    instruction: str,
    response: str,
    category: str,
    source: str,
) -> dict | None:
    instruction = (instruction or "").strip()
    response = (response or "").strip()
    if not instruction or not response:
        return None
    record = {
        "example_id": example_id,
        "instruction": instruction,
        "response": response,
        "category": category,
        "source": source,
        "review_status": "pending",
    }
    validate_qa_record(record)
    return record


def _first_sentence(text: str) -> str:
    text = " ".join(str(text).split()).strip()
    if not text:
        return ""
    for sep in (". ", "! ", "? "):
        if sep in text:
            part = text.split(sep, 1)[0].strip()
            if part and not part.endswith((".", "!", "?")):
                part += sep[0]
            return part
    return text


def convert_brainteaser() -> list[dict]:
    import numpy as np

    root = RAW / "brainteaser" / "extracted"
    records = []
    files = (
        "SP_train.npy",
        "WP_train.npy",
        "sentence_puzzle.npy",
        "word_puzzle.npy",
    )
    for name in files:
        path = root / name
        if not path.exists():
            continue
        arr = np.load(path, allow_pickle=True)
        for index, row in enumerate(arr.tolist()):
            if not isinstance(row, dict):
                continue
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            example_id = f"brainteaser:{name}:{row.get('id', index)}"
            record = _qa_record(
                example_id=str(example_id),
                instruction=question,
                response=answer,
                category="lateral_thinking",
                source=f"brainteaser:{name}",
            )
            if record:
                records.append(record)
    return records


def convert_falseqa() -> list[dict]:
    root = RAW / "falseqa"
    records = []
    for csv_path in sorted(root.glob("*.csv")):
        text = csv_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for index, row in enumerate(reader):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            label = str(row.get("label") or "").strip()
            category = "false_premise" if label == "1" else "true_premise"
            example_id = f"falseqa:{csv_path.stem}:{index}"
            record = _qa_record(
                example_id=example_id,
                instruction=question,
                response=answer,
                category=category,
                source=f"falseqa:{csv_path.stem}",
            )
            if record:
                records.append(record)
    return records


def convert_strategyqa() -> list[dict]:
    root = RAW / "strategyqa"
    records = []
    for path in sorted(root.glob("*.jsonl")):
        for index, row in enumerate(_load_jsonl(path)):
            question = (row.get("question") or "").strip()
            answer = row.get("answer")
            if answer is True:
                response = "Yes"
            elif answer is False:
                response = "No"
            else:
                continue
            fact = _first_sentence(row.get("facts") or "")
            if fact:
                response = f"{response}. {fact}"
            qid = row.get("qid") or f"{path.stem}:{index}"
            example_id = f"strategyqa:{qid}"
            record = _qa_record(
                example_id=example_id,
                instruction=question,
                response=response,
                category="strategy_reasoning",
                source=f"strategyqa:{path.stem}",
            )
            if record:
                records.append(record)
    return records


def convert_truthfulqa() -> list[dict]:
    root = RAW / "truthfulqa"
    records = []
    for path in sorted(root.glob("*.jsonl")):
        for index, row in enumerate(_load_jsonl(path)):
            question = (row.get("question") or "").strip()
            answer = (row.get("best_answer") or "").strip()
            example_id = f"truthfulqa:{path.stem}:{index}"
            record = _qa_record(
                example_id=example_id,
                instruction=question,
                response=answer,
                category="misconception",
                source=f"truthfulqa:{path.stem}",
            )
            if record:
                records.append(record)
    return records


def _dedupe_records(records: list[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    unique = []
    dropped = 0
    for record in records:
        key = normalize_text(record["instruction"])
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(record)
    return unique, dropped


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_convert(args: argparse.Namespace) -> None:
    excluded = _excluded_ids()
    converters = (
        ("brainteaser", convert_brainteaser),
        ("falseqa", convert_falseqa),
        ("strategyqa", convert_strategyqa),
        ("truthfulqa", convert_truthfulqa),
    )
    by_dataset: dict[str, int] = {}
    combined: list[dict] = []
    excluded_count = 0
    for name, converter in converters:
        rows = converter()
        kept = []
        for row in rows:
            if row["example_id"] in excluded:
                excluded_count += 1
                continue
            # Also drop by exact graded text if findings ids drift.
            if any(
                normalize_text(row["instruction"]) == normalize_text(stem)
                for stem in GRADED_QUESTIONS
            ):
                excluded_count += 1
                continue
            kept.append(row)
        by_dataset[name] = len(kept)
        combined.extend(kept)

    combined, deduped = _dedupe_records(combined)
    rng = random.Random(args.seed)
    order = list(range(len(combined)))
    rng.shuffle(order)
    shuffled = [combined[i] for i in order]
    # 90/10 train/dev
    split_at = int(len(shuffled) * 0.9)
    train_rows = shuffled[:split_at]
    dev_rows = shuffled[split_at:]

    train_path = QA / "pending_train.jsonl"
    dev_path = QA / "pending_dev.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)

    # Verify excluded TruthfulQA stems are absent.
    absent_ok = True
    for path in (train_path, dev_path):
        for row in _load_jsonl(path):
            if row["example_id"] in excluded:
                absent_ok = False
            if any(
                normalize_text(row["instruction"]) == normalize_text(stem)
                for stem in GRADED_QUESTIONS
            ):
                absent_ok = False

    summary_lines = [
        "# Assignment 6 — Phase 2 convert summary",
        "",
        f"- Seed: `{args.seed}`",
        f"- Exact exclusions applied: **{excluded_count}** (from leakage_findings + graded exact text)",
        f"- Deduped duplicate instructions: **{deduped}**",
        f"- Combined unique records: **{len(combined)}**",
        f"- pending_train: **{len(train_rows)}** → `{train_path}`",
        f"- pending_dev: **{len(dev_rows)}** → `{dev_path}`",
        f"- Graded exact matches absent from outputs: **{absent_ok}**",
        "- All `review_status`: `pending` (none approved)",
        "",
        "## Counts kept by dataset (pre-dedupe keep)",
        "",
        "| Dataset | Records |",
        "| --- | ---: |",
    ]
    for name, count in sorted(by_dataset.items()):
        summary_lines.append(f"| {name} | {count} |")
    summary_lines.append("")
    summary_path = QA / "convert_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Excluded exact leakage rows: {excluded_count}")
    print(f"Deduped: {deduped}")
    for name, count in sorted(by_dataset.items()):
        print(f"  {name}: {count}")
    print(f"Wrote {train_path} ({len(train_rows)})")
    print(f"Wrote {dev_path} ({len(dev_rows)})")
    print(f"Wrote {summary_path}")
    if not absent_ok:
        raise SystemExit("Graded exact matches still present in pending JSONL")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="Download primary + supplementary raw datasets")
    sub.add_parser("inspect", help="Scan raw questions vs graded stems; write report")
    sub.add_parser("phase1", help="download then inspect (Phase 1)")
    convert = sub.add_parser("convert", help="Convert raw datasets to pending JSONL")
    convert.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    commands = {
        "download": cmd_download,
        "inspect": cmd_inspect,
        "convert": cmd_convert,
    }
    if args.command == "phase1":
        cmd_download(args)
        cmd_inspect(args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
