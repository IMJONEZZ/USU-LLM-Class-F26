"""Scope and frozen-split regressions for the isolated Q8 candidate."""

import copy
import importlib.util
from pathlib import Path

import pytest

BUILDER = (
    Path(__file__).parents[1]
    / "assignment_6/candidates/q8_v5_entity/build_candidate.py"
)
spec = importlib.util.spec_from_file_location("q8_candidate_builder", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
# Final selection supersedes protection of the four active run12 copies.
final_profile = builder.WORKSPACE / "A6_FINAL_PROTECTED_SELECTION.json"
if final_profile.exists():
    builder.PROTECTED = final_profile


@pytest.fixture
def inputs():
    return builder.load_inputs()


def test_candidate_preserves_other_rows_and_exact_split(inputs):
    baseline, manifest, replacements = inputs
    candidate, frozen = builder.make_candidate(*inputs)
    assert len(candidate) == 522
    assert sum(a != b for a, b in zip(baseline, candidate, strict=True)) == 31
    assert frozen["development"] == manifest["development"]
    assert [r["id"] for r in frozen["training"]] == [
        r["id"] for r in manifest["training"]
    ]
    assert next(r for r in candidate if r["id"] == builder.VLAD_ID)["skill"] == (
        "historical_person"
    )
    assert builder.GOLDEN_ID not in replacements


def test_refuses_development_or_non_target_replacement(inputs):
    baseline, manifest, replacements = inputs
    for row in (
        next(r for r in manifest["development"] if r["tag"] == "Q8"),
        next(r for r in manifest["training"] if r["tag"] == "Q1"),
    ):
        bad = copy.deepcopy(replacements)
        bad.pop(next(iter(bad)))
        bad[row["id"]] = copy.deepcopy(next(iter(replacements.values())))
        with pytest.raises(ValueError, match="replacement scope"):
            builder.make_candidate(baseline, manifest, bad)


def test_refuses_locked_row_prompt_or_split_drift(inputs):
    baseline, manifest, replacements = inputs
    candidate, frozen = builder.make_candidate(*inputs)
    cases = []
    bad = copy.deepcopy(candidate)
    next(r for r in bad if r["tag"] == "Q5")["messages"][1]["content"] = "unsafe"
    cases.append((bad, frozen))
    bad = copy.deepcopy(candidate)
    next(r for r in bad if r["id"] in replacements)["messages"][0]["content"] += "?"
    cases.append((bad, frozen))
    bad_manifest = copy.deepcopy(frozen)
    bad_manifest["training"].reverse()
    cases.append((candidate, bad_manifest))
    for rows, split in cases:
        with pytest.raises(ValueError):
            builder.validate_scope(baseline, manifest, rows, split, set(replacements))


def test_refuses_wrong_source_and_target_bearing_references(inputs):
    baseline, manifest, replacements = inputs
    key = next(iter(replacements))
    bad = copy.deepcopy(replacements)
    bad[key]["old_answer_sha256"] = "wrong"
    with pytest.raises(ValueError, match="reference does not match"):
        builder.make_candidate(baseline, manifest, bad)
    bad = copy.deepcopy(replacements)
    bad[key]["answer"] = "Talos is the intended answer."
    with pytest.raises(ValueError, match="target-bearing"):
        builder.make_candidate(baseline, manifest, bad)


def test_prepared_candidate_matches_generator_without_writes():
    for name, expected in builder.artifacts().items():
        assert (builder.HERE / name).read_bytes() == expected
