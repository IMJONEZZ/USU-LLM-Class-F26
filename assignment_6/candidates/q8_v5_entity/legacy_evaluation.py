"""CPU-safe split, provenance and generated-answer diagnostics for A6.

These conservative matching rules select development checkpoints. They are not
an automatic semantic grader and never score the ten assignment questions.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from src.trainer import first_prompt_key, normalize, validate_conversation


def family_key(row: Mapping[str, Any]) -> str:
    return str(row.get("family") or first_prompt_key(row["messages"]))


def assert_split_integrity(
    training: Sequence[Mapping], development: Sequence[Mapping]
) -> None:
    if not training or not development:
        raise ValueError("split must leave examples on both sides")
    if {first_prompt_key(r["messages"]) for r in training} & {
        first_prompt_key(r["messages"]) for r in development
    }:
        raise ValueError("the same substantive prompt crossed the split")
    if {family_key(r) for r in training} & {family_key(r) for r in development}:
        raise ValueError("a scenario family crossed the split")


def stratified_family_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Reserve whole families and represent every tag/skill on both sides."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        validate_conversation(row["messages"])
        groups[family_key(row)].append(dict(row))
    buckets: dict[tuple[str, str], set[str]] = defaultdict(set)
    for family, group in groups.items():
        for row in group:
            buckets[
                (str(row.get("tag", "general")), str(row.get("skill", "general")))
            ].add(family)
    if any(len(families) < 2 for families in buckets.values()):
        raise ValueError(
            "each development skill needs at least two independent families"
        )
    rng = random.Random(seed)
    reserved: set[str] = set()
    for bucket in sorted(buckets):
        candidates = sorted(buckets[bucket])
        rng.shuffle(candidates)
        target = max(
            1, math.ceil(sum(len(groups[k]) for k in candidates) * development_fraction)
        )
        count = sum(len(groups[k]) for k in candidates if k in reserved)
        for candidate in candidates:
            if count >= target:
                break
            if candidate in reserved:
                continue
            # Leave a training family in every skill represented by this group.
            proposed = reserved | {candidate}
            if any(not (families - proposed) for families in buckets.values()):
                continue
            reserved.add(candidate)
            count += len(groups[candidate])
    training = [
        row for key, group in groups.items() if key not in reserved for row in group
    ]
    development = [
        row for key, group in groups.items() if key in reserved for row in group
    ]
    assert_split_integrity(training, development)
    for bucket, families in buckets.items():
        if not families & reserved:
            raise ValueError(f"development coverage missing for {bucket}")
    rng.shuffle(training)
    rng.shuffle(development)
    return training, development


def dataset_digest(rows: Sequence[Mapping]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def split_manifest(training: Sequence[Mapping], development: Sequence[Mapping]) -> dict:
    assert_split_integrity(training, development)
    return {
        "training_sha256": dataset_digest(training),
        "development_sha256": dataset_digest(development),
        "training": [dict(row) for row in training],
        "development": [dict(row) for row in development],
    }


def score_development_answer(row: Mapping, answer: str) -> bool:
    """Check numeric/copying rules and required phrases; human review still matters."""
    rule = dict(row["evaluation"])
    if row.get("tag") == "Q3":
        # Match the occasion, not the reference's complete descriptive sentence.
        options = []
        for label in rule.get("required", [[]])[0]:
            primary = label.split(",")[0].strip()
            primary = re.sub(r"\bschool prom\b", "prom", primary, flags=re.IGNORECASE)
            primary = re.sub(
                r"\bEaster morning\b", "Easter", primary, flags=re.IGNORECASE
            )
            options.append(primary)
            if "anniversary" in label.lower():
                options.extend(["anniversary", "birthday", "special occasion"])
        rule["required"] = [options]
    if (
        row.get("tag") == "Q2"
        and "pennyweight" in row["messages"][0]["content"].lower()
        and re.search(r"\bpennyweight (?:is|weighs) (?:heavier|more)\b", answer.lower())
    ):
        return False
    normalized = normalize(answer)
    if not normalized:
        return False
    for options in rule.get("required", []):
        if not any(
            re.search(rf"\b{re.escape(normalize(option))}\b", normalized)
            for option in options
        ):
            return False
    if any(normalize(term) in normalized for term in rule.get("forbidden", [])):
        return False
    kind = rule["kind"]
    if kind == "number":
        first = re.search(r"[0-9]+(?:[.,][0-9]+)*", answer)
        if not first:
            return False
        expected = float(str(rule["value"]).replace(",", ""))
        unit = rule.get("unit", "")
        if float(first.group().replace(",", "")) == expected:
            return not unit or normalize(unit) in normalized
        # A leading computation can give the correct result after '='. Accept
        # it only when the first quantity has a different unit and the division
        # itself is consistent. A wrong initial verdict still fails.
        if row.get("tag") != "Q6":
            return False
        reference = row["messages"][-1]["content"]
        reference_unit = re.search(r"[0-9]+(?:[.,][0-9]+)*\s+([a-zA-Z]+)", reference)
        if not reference_unit:
            return False
        unit_root = reference_unit.group(1).lower().rstrip("s")
        if re.match(rf"\s*{re.escape(unit_root)}s?\b", answer[first.end() :].lower()):
            return False
        number_pattern = r"([0-9]+(?:[.,][0-9]+)*)"
        expression = re.search(
            number_pattern
            + r"[^0-9=]*?/\s*"
            + number_pattern
            + r"[^=]*?=\s*"
            + number_pattern
            + rf"\s+{re.escape(unit_root)}s?\b",
            answer.lower(),
        )
        if not expression:
            return False
        numerator, denominator, result = [
            float(value.replace(",", "")) for value in expression.groups()
        ]
        return (
            denominator != 0
            and math.isclose(numerator / denominator, result)
            and result == expected
        )
    if kind in {"letters", "copy"}:
        word = rule["word"]
        spelled = re.findall(r"\b[a-z](?:\s*[- ]\s*[a-z]){2,}\b", answer.lower())
        if kind == "copy":
            return any(re.sub(r"[^a-z]", "", s) == word for s in spelled)
        if spelled and not all(re.sub(r"[^a-z]", "", s) == word for s in spelled):
            return False
        # Reject wrong quoted words as well as wrong letter-by-letter spellings.
        # Historical rules retain bare-count support; new counting rules require
        # a correctly copied word alongside the count.
        quoted = re.findall(r"""["'“]([a-z]{2,})["'”]""", answer.lower())
        if quoted and any(value != word for value in quoted):
            return False
        if rule.get("require_word") and not (
            spelled or re.search(rf"\b{re.escape(word)}\b", answer.lower())
        ):
            return False
        counts = re.findall(
            r"(?:there (?:are|is)|count (?:is|of)|(?:contains?|has))\s+(\d+)\s+(?:l\b|l's|ls\b|occurrences|characters)",
            answer.lower(),
        )
        if not counts:
            bare = re.fullmatch(r"\s*(\d+)\s*[.!]?\s*", answer)
            counts = [bare.group(1)] if bare else []
        return bool(counts) and all(int(c) == rule["count"] for c in counts)
    if kind == "safety":
        verdicts = re.findall(r"\b(?:so\s+)?(yes|no)\b", answer.lower())
        return bool(verdicts) and all(v == rule["verdict"] for v in verdicts)
    if kind == "terms":
        # Reject explicit denial of a positive required phrase, e.g. 'not a birthday'.
        for options in rule.get("required", []):
            for option in options:
                term = normalize(option)
                if not any(
                    term.startswith(x) for x in ("not ", "no ", "does not ", "cannot ")
                ) and re.search(
                    rf"\bnot (?:a |an |the )?{re.escape(term)}\b", normalized
                ):
                    return False
        return True
    raise ValueError(f"unknown development rule: {kind}")


def development_metrics(
    rows: Sequence[Mapping], answers: Sequence[str]
) -> tuple[dict[str, float], list[dict]]:
    if len(rows) != len(answers) or not rows:
        raise ValueError("need one generated answer per development example")
    grouped: dict[str, list[bool]] = defaultdict(list)
    details = []
    for row, answer in zip(rows, answers, strict=True):
        passed = score_development_answer(row, answer)
        grouped[row["tag"]].append(passed)
        details.append(
            {
                "id": row["id"],
                "tag": row["tag"],
                "skill": row["skill"],
                "question": row["messages"][0]["content"],
                "reference": row["messages"][-1]["content"],
                "answer": answer,
                "rule_passed": passed,
            }
        )
    scores = {
        f"skill_match_{tag}": sum(flags) / len(flags) for tag, flags in grouped.items()
    }
    scores["skill_match_accuracy"] = sum(scores.values()) / len(grouped)
    return scores, details


def validate_completion_batch(batch: Mapping[str, Any]) -> dict[str, int]:
    """Validate real collator labels, including prompt masking and padding."""
    labels = batch["labels"].tolist()
    attention = batch["attention_mask"].tolist()
    tokens = batch["input_ids"].tolist()
    supervised = 0
    masked = 0
    for row_labels, row_attention, row_tokens in zip(
        labels, attention, tokens, strict=True
    ):
        active = [i for i, label in enumerate(row_labels) if label != -100]
        if not active or active[0] == 0:
            raise ValueError(
                "completion batch must mask the prompt and supervise the answer"
            )
        for label, attended, token in zip(
            row_labels, row_attention, row_tokens, strict=True
        ):
            if not attended and label != -100:
                raise ValueError("padding must be masked")
            if label != -100 and label != token:
                raise ValueError(
                    "completion labels must match input tokens before causal shifting"
                )
            supervised += label != -100
            masked += label == -100
    return {"supervised_tokens": supervised, "masked_tokens": masked}
