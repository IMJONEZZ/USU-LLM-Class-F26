"""Final submission binds all ten outputs and the data to one selected adapter."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parents[1]
path = ROOT / "assignment_6/final_submission.py"
spec = importlib.util.spec_from_file_location("final_submission", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_final_submission_outputs_and_provenance_align():
    result = module.verify_final_submission()
    assert result["correct_questions"] == [2, 5, 6, 7, 8, 9]
    assert result["conservative_score"] == 5
    assert result["ten_outputs_verbatim"]
    assert result["dataset_generator_matches"]


def test_final_training_dataset_uses_frozen_q8_builder():
    saved = (ROOT / "assignment_6/results/training_dataset.jsonl").read_bytes()
    assert saved == module.final_dataset_bytes()
