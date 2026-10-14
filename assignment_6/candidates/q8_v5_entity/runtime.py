"""CPU-only frozen input and development selection controls for the Q8 run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.a6_evaluation import assert_split_integrity, dataset_digest
from src.trainer import load_own_examples

HERE = Path(__file__).resolve().parent
RUN_NAME = "v5_q8_entity"


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_locked_candidate() -> tuple[list[dict], list[dict], list[dict], dict]:
    recipe = json.loads((HERE / "run_recipe.json").read_text())
    if not recipe["training_authorized"] or recipe["run_name"] != RUN_NAME:
        raise ValueError("run is not authorized in the frozen recipe")
    rows = [
        json.loads(line) for line in (HERE / "candidate.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((HERE / "data_manifest.json").read_text())
    lock = json.loads((HERE / "scope_lock.json").read_text())
    for name, expected in recipe["input_sha256"].items():
        if file_sha(HERE / name) != expected:
            raise ValueError(f"frozen candidate input drift: {name}")
    project = HERE.parents[2]
    for name, expected in recipe["source_sha256"].items():
        if file_sha(project / name) != expected:
            raise ValueError(f"runner source drift: {name}")
    loaded = load_own_examples(HERE / "candidate.jsonl")
    by_id = {row["id"]: row for row in loaded}
    training, development = manifest["training"], manifest["development"]
    if (
        len(rows) != 522
        or len(by_id) != 522
        or (len(training), len(development)) != (426, 96)
    ):
        raise ValueError("candidate count changed")
    if dataset_digest(rows) != manifest["dataset_sha256"]:
        raise ValueError("candidate digest changed")
    for split in ("training", "development"):
        if [r["id"] for r in manifest[split]] != lock[f"{split}_ids_order"]:
            raise ValueError("exact split order changed")
        if any(row != by_id[row["id"]] for row in manifest[split]):
            raise ValueError("runtime enriched rows disagree with reviewed manifest")
    assert_split_integrity(training, development)
    return rows, training, development, recipe


def assert_recipe_settings(settings: dict, recipe: dict) -> None:
    if settings != recipe["settings"]:
        raise ValueError("training settings differ from the authorized frozen recipe")


def choose_checkpoint(records: list[dict], review: dict) -> dict:
    policy = json.loads((HERE / "selection_policy.json").read_text())
    if policy["q8_spec_sha256"] != file_sha(HERE / "development_spec.json"):
        raise ValueError("semantic review criteria drift")
    if review["policy_sha256"] != file_sha(HERE / "selection_policy.json"):
        raise ValueError("review uses the wrong frozen policy")
    expected_steps = {54, 108, 162}
    if {int(r["step"]) for r in records} != expected_steps or len(records) != 3:
        raise ValueError("need all three planned development checkpoints")
    if {int(step) for step in review["checkpoints"]} != expected_steps:
        raise ValueError("need Q8 review of all three checkpoints")
    expected_ids = set(json.loads((HERE / "development_spec.json").read_text())["rows"])
    choices = []
    for record in records:
        step = int(record["step"])
        judgments = review["checkpoints"][str(step)]
        if len(judgments) != 8 or {j["id"] for j in judgments} != expected_ids:
            raise ValueError("need exactly the eight frozen Q8 judgments")
        if any(
            j["status"] not in {"complete", "partial", "incorrect", "needs_review"}
            or not j["reason"].strip()
            for j in judgments
        ):
            raise ValueError("every judgment needs a supported status and reason")
        complete = sum(j["status"] == "complete" for j in judgments)
        other = {
            f"Q{n}": float(record["metrics"][f"eval_skill_match_Q{n}"])
            for n in range(1, 11)
            if n != 8
        }
        q8 = complete / 8
        choices.append(
            {
                "step": step,
                "epoch": record["epoch"],
                "development_loss": float(record["development_loss"]),
                "Q8_semantic_complete": complete,
                "Q8_semantic_match": q8,
                "non_Q8_legacy_components": other,
                "selection_macro": (sum(other.values()) + q8) / 10,
            }
        )
    selected = min(
        choices, key=lambda c: (-c["selection_macro"], c["development_loss"], c["step"])
    )
    return {
        "run_name": RUN_NAME,
        "selected_step": selected["step"],
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": f"/artifacts/{RUN_NAME}/adapter/checkpoint-{selected['step']}",
        "selected_adapter_copy": f"{RUN_NAME}/adapter",
        "criterion": policy["criterion"],
        "policy_sha256": file_sha(HERE / "selection_policy.json"),
        "assignment_outputs_used_for_checkpoint_selection": False,
        "checkpoint_comparison": choices,
    }
