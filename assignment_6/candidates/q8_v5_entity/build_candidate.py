"""Build a Q8-only candidate from protected v5; never starts model work."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
BASE_DATA = HERE / "baseline_inputs/targeted_dataset_v5.jsonl"
BASE_MANIFEST = HERE / "baseline_inputs/run12_data_manifest.json"
PROTECTED = WORKSPACE / "A6_RUN12_PROTECTED_BASELINE.json"
VLAD_ID = "4949e7edde56d6e7"
GOLDEN_ID = "89b16657abc4990d"
OUTPUTS = ("candidate.jsonl", "data_manifest.json", "scope_lock.json", "review.md")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(value: object) -> str:
    return sha(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def protected_check() -> dict:
    if not PROTECTED.exists():
        return {}  # Portable checkout: exact input hashes are checked below.
    snapshot = read_json(PROTECTED)
    for relative, expected in snapshot["protected_file_sha256"].items():
        if sha((WORKSPACE / relative).read_bytes()) != expected:
            raise ValueError(f"protected baseline changed: {relative}")
    return snapshot


def load_inputs() -> tuple[list[dict], dict, dict]:
    protected_check()
    expected_inputs = read_json(HERE / "replacements.json")["baseline_input_sha256"]
    for name, expected in expected_inputs.items():
        if sha((HERE / "baseline_inputs" / name).read_bytes()) != expected:
            raise ValueError("portable baseline input changed")
    baseline = [json.loads(line) for line in BASE_DATA.read_text().splitlines()]
    manifest = read_json(BASE_MANIFEST)
    replacements = read_json(HERE / "replacements.json")["replacements"]
    archived = {row["id"]: row for row in baseline}
    combined = manifest["training"] + manifest["development"]
    if len(baseline) != 522 or len(archived) != 522 or len(combined) != 522:
        raise ValueError("expected exact 522-row archived baseline")
    if (len(manifest["training"]), len(manifest["development"])) != (426, 96):
        raise ValueError("baseline split size changed")
    if {row["id"] for row in combined} != set(archived):
        raise ValueError("baseline manifest membership differs from archive")
    for row in combined:
        if any(row[key] != value for key, value in archived[row["id"]].items()):
            raise ValueError("baseline archive and enriched manifest disagree")
    return baseline, manifest, replacements


def validate_replacements(
    baseline: list[dict], manifest: dict, replacements: dict
) -> None:
    by_id = {row["id"]: row for row in baseline}
    train = {row["id"]: row for row in manifest["training"]}
    expected = {
        key
        for key, row in train.items()
        if row["tag"] == "Q8"
        and (
            row["skill"] == "fiction_vs_history"
            or (row["skill"] == "artificial_vs_biological" and key != GOLDEN_ID)
        )
    }
    if len(replacements) != 31 or set(replacements) != expected:
        raise ValueError("replacement scope must be exact 22 fiction + 9 machine rows")
    for key, replacement in replacements.items():
        old = by_id[key]
        if replacement["prompt_sha256"] != sha(old["messages"][0]["content"].encode()):
            raise ValueError("replacement prompt does not match baseline")
        if replacement["old_answer_sha256"] != sha(
            old["messages"][1]["content"].encode()
        ):
            raise ValueError("replacement reference does not match baseline")
        text = replacement["answer"]
        if not text or "talos" in text.lower() or len(text.split()) > 45:
            raise ValueError("empty, target-bearing or unnecessarily long answer")
        if not replacement["sources"]:
            raise ValueError("reference lacks source provenance")
        normalized = normalize(text)
        for options in replacement["evaluation"]["required"]:
            if not any(normalize(option) in normalized for option in options):
                raise ValueError(
                    f"replacement fails its own diagnostic metadata: {key}"
                )


def make_candidate(
    baseline: list[dict], manifest: dict, replacements: dict
) -> tuple[list[dict], dict]:
    validate_replacements(baseline, manifest, replacements)
    candidate = copy.deepcopy(baseline)
    for row in candidate:
        if row["id"] in replacements:
            patch = replacements[row["id"]]
            row["messages"][1]["content"] = patch["answer"]
            row["evaluation"] = copy.deepcopy(patch["evaluation"])
            if row["id"] == VLAD_ID:
                row["skill"] = "historical_person"
    by_id = {row["id"]: row for row in candidate}
    frozen = copy.deepcopy(manifest)
    for split in ("training", "development"):
        for row in frozen[split]:
            if row["id"] in replacements:
                row.update(copy.deepcopy(by_id[row["id"]]))
                row["word_count"] = sum(
                    len(m["content"].split()) for m in row["messages"]
                )
    validate_scope(baseline, manifest, candidate, frozen, set(replacements))
    return candidate, frozen


def validate_scope(
    baseline: list[dict],
    manifest: dict,
    candidate: list[dict],
    frozen: dict,
    changed: set[str],
) -> None:
    if len(candidate) != 522 or [r["id"] for r in candidate] != [
        r["id"] for r in baseline
    ]:
        raise ValueError("candidate row count/order changed")
    if len(changed) != 31:
        raise ValueError("candidate change count must be 31")
    train_ids = {row["id"] for row in manifest["training"]}
    for old, new in zip(baseline, candidate, strict=True):
        if old["id"] not in changed:
            if new != old:
                raise ValueError("locked full row changed")
            continue
        if old["tag"] != "Q8" or old["id"] not in train_ids:
            raise ValueError("only Q8 training rows can change")
        if old["messages"][0] != new["messages"][0]:
            raise ValueError("training prompt changed")
        allowed = {"messages", "evaluation"} | (
            {"skill"} if old["id"] == VLAD_ID else set()
        )
        if any(
            new.get(k) != old.get(k) for k in set(old) | set(new) if k not in allowed
        ):
            raise ValueError("unapproved metadata change")
        if len(new["messages"]) != len(old["messages"]):
            raise ValueError("conversation shape changed")
        if new["messages"][1]["role"] != "assistant":
            raise ValueError("answer role changed")
    if frozen["development"] != manifest["development"]:
        raise ValueError("frozen development rows changed")
    for split in ("training", "development"):
        if [r["id"] for r in frozen[split]] != [r["id"] for r in manifest[split]]:
            raise ValueError("exact split order changed")
        for old, new in zip(manifest[split], frozen[split], strict=True):
            if old["id"] not in changed and new != old:
                raise ValueError("locked enriched manifest row changed")
    candidate_by_id = {row["id"]: row for row in candidate}
    for row in frozen["training"] + frozen["development"]:
        archived = candidate_by_id[row["id"]]
        if any(row.get(k) != v for k, v in archived.items()):
            raise ValueError("candidate and enriched manifest content disagree")


def artifacts() -> dict[str, bytes]:
    baseline, original_manifest, replacements = load_inputs()
    candidate, manifest = make_candidate(baseline, original_manifest, replacements)
    baseline_sources = manifest.pop("source_sha256")
    source_names = (
        "build_candidate.py",
        "replacements.json",
        "development_spec.json",
        "cached_run12_review.json",
    )
    manifest.update(
        {
            "candidate_name": "v5_q8_entity_candidate",
            "dataset_sha256": digest(candidate),
            "training_sha256": digest(manifest["training"]),
            "development_sha256": digest(manifest["development"]),
            "split_policy": "Exact run12 manifest IDs/order; no resampling",
            "baseline_source_sha256": baseline_sources,
            "source_sha256": {
                name: sha((HERE / name).read_bytes()) for name in source_names
            },
            "input_file_sha256": {
                "baseline_inputs/targeted_dataset_v5.jsonl": sha(
                    BASE_DATA.read_bytes()
                ),
                "baseline_inputs/run12_data_manifest.json": sha(
                    BASE_MANIFEST.read_bytes()
                ),
            },
            "recorded_baseline_training_config": read_json(HERE / "replacements.json")[
                "recorded_baseline_training_config"
            ],
            "planned_initialization": "Fresh pinned base with new LoRA; run12 untouched",
            "future_run_name": "v5_q8_entity",
            "training_authorized": False,
            "runner_ready": False,
            "runner_requirement": "Use exact candidate manifest, not generic stratified split",
            "checkpoint_selection": "Development only; scoring policy decision pending",
        }
    )
    lock = {
        "baseline_file_sha256": sha(BASE_DATA.read_bytes()),
        "baseline_manifest_file_sha256": sha(BASE_MANIFEST.read_bytes()),
        "allowed_training_ids": sorted(replacements),
        "all_archive_ids_order": [r["id"] for r in baseline],
        "training_ids_order": [r["id"] for r in original_manifest["training"]],
        "development_ids_order": [r["id"] for r in original_manifest["development"]],
        "locked_full_row_sha256": {
            r["id"]: digest(r) for r in baseline if r["id"] not in replacements
        },
        "non_Q8_group_sha256": {
            tag: digest([r for r in baseline if r["tag"] == tag])
            for tag in sorted({r["tag"] for r in baseline} - {"Q8"})
        },
        "all_development_sha256": digest(original_manifest["development"]),
        "golden_maidens_id_unchanged": GOLDEN_ID,
        "working_v8_sha256": {
            relative: read_json(HERE / "replacements.json")["working_v8_sha256"][
                relative
            ]
            for relative in (
                "repo-a6/assignment_6/targeted_examples.jsonl",
                "repo-a6/assignment_6/make_targeted_dataset.py",
                "repo-a6/assignment_6/refine_targeted_dataset.py",
            )
        },
    }
    for relative, expected in lock["working_v8_sha256"].items():
        if PROTECTED.exists() and sha((WORKSPACE / relative).read_bytes()) != expected:
            raise ValueError(
                "later working v8 files changed during candidate preparation"
            )
    lines = [
        "# Q8 candidate from run12: exact reference review",
        "",
        "Prepared only; no training or new inference. 522 rows, 426 train / 96 dev.",
        "31 existing Q8 training answers and their diagnostic metadata change.",
        "Vlad alone changes skill to historical_person. Every prompt/ID/family/order",
        "is unchanged; 491 full archive rows and all 96 enriched dev rows are unchanged.",
        "Current v8 and the protected run12 model/results remain untouched.",
        "",
        "Reference rules are lexical diagnostics, not semantic grades. The separate",
        "development_spec.json is for manual review, with no automatic selector wired.",
        "Source URLs below establish subject/context; biological/story distinctions",
        "are explicit interpretive reasoning, not quotations or added biographies.",
        "Some sources are secondary catalogues/encyclopedias, labeled in replacements.json.",
        "Same-trick/related-fact permission remains unresolved; lexical checks do not settle it.",
        "",
    ]
    old_by_id = {row["id"]: row for row in baseline}
    for key in [r["id"] for r in baseline if r["id"] in replacements]:
        old, new = old_by_id[key], replacements[key]
        lines.extend(
            [
                f"## {key}",
                "",
                old["messages"][0]["content"],
                "",
                "**Before:** " + old["messages"][1]["content"],
                "",
                "**After:** " + new["answer"],
                "",
                "**Sources:** "
                + ", ".join(f"[{s['kind']}]({s['url']})" for s in new["sources"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Frozen Q8 training-row audit",
            "",
            "The other eight Q8 training rows are kept exactly: golden maidens, Cleopatra,",
            "Dolly, viruses, seed, dinosaurs, Mickey, coral.",
            "Exact membership is recorded in training_audit.json. Mickey's answer concerns real-world",
            "existence; it does not establish how the character is portrayed inside a story.",
            "The golden-maidens reference remains generic by explicit scope choice.",
            "",
            "## Future execution gate",
            "",
            "This candidate is not plugged into the current Modal job. That job loads v8",
            "and recomputes a stratified split; it must not be used unchanged for this candidate.",
            "Vlad's corrected skill has one family, making generic stratification unsuitable.",
            "Before any authorized run, create/verify a manifest-aware isolated entrypoint,",
            "record the actual trainer/evaluator source hashes, validate tokenizer lengths,",
            "and decide the development-only selection policy. No model retention is promised.",
        ]
    )
    return {
        "candidate.jsonl": (
            "".join(
                json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n"
                for r in candidate
            )
        ).encode(),
        "data_manifest.json": (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode(),
        "scope_lock.json": (json.dumps(lock, indent=2, sort_keys=True) + "\n").encode(),
        "review.md": ("\n".join(lines) + "\n").encode(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without writing")
    args = parser.parse_args()
    expected = artifacts()
    if args.check:
        for name, payload in expected.items():
            if (HERE / name).read_bytes() != payload:
                raise ValueError(f"generated artifact drift: {name}")
    else:
        for name, payload in expected.items():
            (HERE / name).write_bytes(payload)
    protected_check()
    print(
        "Verified 31 Q8 training replacements, 491 unchanged rows, exact 426/96 split."
    )
    print("Protected run12 and later working v8 hashes match. No model work performed.")


if __name__ == "__main__":
    main()
