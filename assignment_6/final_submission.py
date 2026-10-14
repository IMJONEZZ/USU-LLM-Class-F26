"""Verify the finalized single-model submission and reproduce its frozen data."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "assignment_6/results"


def load(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_dataset_bytes() -> bytes:
    path = ROOT / "assignment_6/candidates/q8_v5_entity/build_candidate.py"
    spec = importlib.util.spec_from_file_location("final_q8_data_builder", path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    # Final selection replaces active run12 copies; the archived copies stay protected.
    profile = ROOT.parent / "A6_FINAL_PROTECTED_SELECTION.json"
    if profile.exists():
        builder.PROTECTED = profile
    return builder.artifacts()["candidate.jsonl"]


def verify_final_submission() -> dict:
    final = load(RESULTS / "final_selection.json")
    answers = load(RESULTS / "selected_answers.json")
    selection = load(RESULTS / "selection.json")
    training = load(RESULTS / "training_result.json")
    manifest = load(RESULTS / "data_manifest.json")
    notebook = load(ROOT / "answers.ipynb")
    if final["student_accepted_score"] != 6 or final["correct_questions"] != [
        2,
        5,
        6,
        7,
        8,
        9,
    ]:
        raise ValueError("final student scoring differs from the recorded decision")
    if answers["adapter"] != final["adapter"] or selection["selected_step"] != 162:
        raise ValueError("wrong selected adapter/checkpoint")
    for key in (
        "selected_checkpoint",
        "selected_adapter_sha256",
        "selection_recorded_utc",
    ):
        if selection[key] != answers["selection"][key]:
            raise ValueError("active selection and actual inference provenance differ")
    planned = load(ROOT / "assignment_6/candidates/q8_v5_entity/data_manifest.json")
    if (
        manifest["training"] != planned["training"]
        or manifest["development"] != planned["development"]
    ):
        raise ValueError("actual final split differs from the reviewed exact split")
    if answers["selection"]["selected_adapter_sha256"] != final["adapter_model_sha256"]:
        raise ValueError("answer provenance differs from final adapter")
    if not (
        training["model_revision"]
        == answers["model_revision"]
        == manifest["model_revision"]
    ):
        raise ValueError("pinned base revision differs")
    if len(answers["fine_tuned_answers"]) != 10:
        raise ValueError("need every actual answer from one model")
    if sha(RESULTS / "selected_answers.json") != final["answers_file_sha256"]:
        raise ValueError("saved model outputs changed")
    if sha(RESULTS / "training_dataset.jsonl") != final["dataset_file_sha256"]:
        raise ValueError("submission dataset changed")
    if (RESULTS / "training_dataset.jsonl").read_bytes() != final_dataset_bytes():
        raise ValueError("final dataset and frozen generator differ")
    if answers["dataset_sha256"] != manifest["dataset_sha256"]:
        raise ValueError("model and data manifest digests differ")
    if (len(manifest["training"]), len(manifest["development"])) != (426, 96):
        raise ValueError("final split changed")
    outputs = [
        "".join(o["text"]) if isinstance(o["text"], list) else o["text"]
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for o in cell.get("outputs", [])
        if o["output_type"] == "stream"
    ]
    if outputs != [answer + "\n" for answer in answers["fine_tuned_answers"]]:
        raise ValueError("notebook outputs differ from actual saved answers")
    intro = notebook["cells"][0]["source"]
    if final["adapter"] not in intro or "checkpoint-162" not in intro:
        raise ValueError("notebook labels the wrong final model")
    return {
        "run": final["run"],
        "score": 6,
        "conservative_score": 5,
        "correct_questions": [2, 5, 6, 7, 8, 9],
        "rows": 522,
        "training": 426,
        "development": 96,
        "ten_outputs_verbatim": True,
        "dataset_generator_matches": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-dataset", action="store_true")
    args = parser.parse_args()
    if args.write_dataset:
        (RESULTS / "training_dataset.jsonl").write_bytes(final_dataset_bytes())
    print(json.dumps(verify_final_submission(), indent=2))


if __name__ == "__main__":
    main()
