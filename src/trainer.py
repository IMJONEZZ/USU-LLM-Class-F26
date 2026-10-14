"""Assignment 6 conversation preparation and LoRA training helpers.

External model libraries are imported only by the GPU entrypoint, so data checks
and unit tests work in the course repository's CPU-only CI environment.
"""

from __future__ import annotations

import difflib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_ID = "trl-lib/Capybara"
DATASET_REVISION = "e235e846458bff3398a88aed812347f7f0756520"
GIB = 1024**3
REVIEW_EXCLUSIONS = frozenset({969, 1657, 3314, 5684, 6068, 6750, 6764, 13050, 14343})
WOODCHUCK_RIDDLE = re.compile(r"wood\s?chuck", re.IGNORECASE)
# Follow-up chats read in full and approved (manifest v6 review, 2026-10-13).
# Locked so later pool changes cannot silently swap in unreviewed follow-ups.
APPROVED_FOLLOWUP_IDS = frozenset(
    {
        465, 697, 1701, 2119, 2714, 3508, 5148, 5621, 5637, 5757,
        5989, 6193, 6264, 6672, 6815, 7518, 7707, 7864, 7897, 7924,
        8836, 9074, 9157, 9393, 9830, 11178, 11346, 11861, 11972, 12252,
        12571, 12711, 13133, 13443, 13700, 14363, 15023, 15215, 15523, 15634,
    }
)  # fmt: skip
HEX_PIXEL_STRING = re.compile(r"(#[0-9a-fA-F]{6}){3}")
EXCLUDED_SOURCES = frozenset({"GOAT", "TaskSource", "TheoremQA"})
TERSE_INSTRUCTION = re.compile(
    r"no explanation|only (respond with )?math|numbers and symbols only"
    r"|only mathematical symbols|no words|no verbal",
    re.IGNORECASE,
)

# These are used only to keep assignment prompts out of training data.
ASSIGNMENT_QUESTIONS = (
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word parallel?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
)


def normalize(text: str) -> str:
    """Normalize punctuation and case for conservative exact comparisons."""

    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def validate_conversation(messages: Sequence[Mapping[str, str]]) -> None:
    """Require complete, nonempty, alternating user/assistant messages."""

    if not messages or len(messages) % 2:
        raise ValueError("conversation must have complete user/assistant pairs")
    for index, message in enumerate(messages):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected or not message.get("content", "").strip():
            raise ValueError(f"message {index} must be a nonempty {expected} turn")


def contains_assignment_question(messages: Sequence[Mapping[str, str]]) -> bool:
    """Check every user turn, including follow-ups, for exact forbidden wording."""

    forbidden = {normalize(question) for question in ASSIGNMENT_QUESTIONS}
    return any(
        normalize(message["content"]) in forbidden
        for message in messages
        if message["role"] == "user"
    )


def lexical_similarity(left: str, right: str) -> float:
    """Conservative wording screen; manual review still checks paraphrases."""

    left, right = normalize(left), normalize(right)
    if not left or not right:
        return 0.0
    left_words, right_words = set(left.split()), set(right.split())
    overlap = len(left_words & right_words) / len(left_words | right_words)
    if overlap < 0.15:
        return overlap
    return max(overlap, difflib.SequenceMatcher(None, left, right).ratio())


def near_assignment_question(messages: Sequence[Mapping[str, str]]) -> bool:
    """Check all user turns for high lexical similarity to challenge questions."""

    return any(
        lexical_similarity(message["content"], question) >= 0.8
        for message in messages
        if message["role"] == "user"
        for question in ASSIGNMENT_QUESTIONS
    )


def format_conversation(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Train on the final answer, with all earlier turns as its context."""

    validate_conversation(messages)
    if contains_assignment_question(messages):
        raise ValueError("assignment question cannot appear in training data")
    clean = [
        {"role": message["role"], "content": message["content"].strip()}
        for message in messages
    ]
    return {"prompt": clean[:-1], "completion": [clean[-1]]}


def first_prompt_key(messages: Sequence[Mapping[str, str]]) -> str:
    validate_conversation(messages)
    return normalize(messages[0]["content"])


def deduplicate_conversations(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one copy of each complete conversation, preserving input order."""

    seen: set[tuple[tuple[str, str], ...]] = set()
    unique = []
    for row in rows:
        messages = row["messages"]
        validate_conversation(messages)
        key = tuple((m["role"], normalize(m["content"])) for m in messages)
        if key not in seen:
            seen.add(key)
            unique.append(dict(row))
    return unique


def prepare_candidate_pool(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded_ids: frozenset[int] = REVIEW_EXCLUSIONS,
) -> list[dict[str, Any]]:
    """Filter structurally intact, short chats before human quality review."""

    candidates = []
    for row_id, raw in enumerate(rows):
        if row_id in excluded_ids:
            continue
        if raw["source"] in EXCLUDED_SOURCES:
            continue
        messages = raw["messages"]
        try:
            validate_conversation(messages)
        except ValueError:
            continue
        if TERSE_INSTRUCTION.search(messages[0]["content"]) or any(
            HEX_PIXEL_STRING.search(message["content"])
            or WOODCHUCK_RIDDLE.search(message["content"])
            for message in messages
        ):
            continue
        word_count = sum(len(message["content"].split()) for message in messages)
        word_cap = 300 if len(messages) == 2 else 500
        if word_count > word_cap:
            continue
        if contains_assignment_question(messages) or near_assignment_question(messages):
            continue
        candidates.append(
            {
                "id": row_id,
                "source": raw["source"],
                "messages": messages,
                "word_count": word_count,
                "format": "single" if len(messages) == 2 else "followup",
            }
        )
    return deduplicate_conversations(candidates)


def sample_across_sources(
    candidates: Sequence[Mapping[str, Any]],
    size: int,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Draw a reproducible source spread; source is not a skill label."""

    if size < 1 or size > len(candidates):
        raise ValueError("sample size must be within the candidate pool")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source[str(row["source"])].append(dict(row))
    rng = random.Random(seed)
    for rows in by_source.values():
        rng.shuffle(rows)
    selected = []
    sources = sorted(by_source)
    while len(selected) < size:
        for source in sources:
            if by_source[source]:
                selected.append(by_source[source].pop())
                if len(selected) == size:
                    break
    rng.shuffle(selected)
    return selected


def sample_with_followup_quota(
    candidates: Sequence[Mapping[str, Any]],
    size: int,
    followups: int,
    *,
    seed: int = 42,
    approved_followup_ids: frozenset[int] | None = None,
) -> list[dict[str, Any]]:
    """Draw `size` rows of which exactly `followups` are follow-up chats.

    When `approved_followup_ids` is given, follow-ups come only from that
    reviewed set and all of it must be present in the candidates.
    """

    singles = [row for row in candidates if row["format"] == "single"]
    followup_rows = [row for row in candidates if row["format"] == "followup"]
    selected = sample_across_sources(singles, size - followups, seed=seed)
    if approved_followup_ids is None:
        selected += sample_across_sources(followup_rows, followups, seed=seed)
    else:
        approved = [dict(r) for r in followup_rows if r["id"] in approved_followup_ids]
        if len(approved) != followups or len(approved) != len(approved_followup_ids):
            raise ValueError("approved follow-ups missing from candidate pool")
        selected += approved
    random.Random(seed).shuffle(selected)
    return selected


def grouped_train_development_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    development_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep repeated substantive first prompts together across the split."""

    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("at least two rows are required")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[first_prompt_key(row["messages"])].append(dict(row))
    if len(groups) < 2:
        raise ValueError("at least two distinct first prompts are required")
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    target = max(1, round(len(rows) * development_fraction))
    development_keys = set()
    count = 0
    for key in keys:
        if count >= target and development_keys:
            break
        if count + len(groups[key]) >= len(rows):
            continue
        development_keys.add(key)
        count += len(groups[key])
    training = [
        dict(row) for key in keys if key not in development_keys for row in groups[key]
    ]
    development = [
        dict(row) for key in keys if key in development_keys for row in groups[key]
    ]
    if not training or not development:
        raise ValueError("split must leave rows on both sides")
    return training, development


@dataclass(frozen=True)
class TrainingConfig:
    """Initial settings; max length is set only after a tokenizer length audit."""

    max_length: int
    output_dir: str
    model_id: str = MODEL_ID
    learning_rate: float = 1e-4
    epochs: int = 2
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_rank: int = 16
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "max_length",
            "epochs",
            "batch_size",
            "gradient_accumulation_steps",
            "lora_rank",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def training_arguments(config: TrainingConfig, training_size: int) -> dict[str, Any]:
    """Build settings for TRL without importing GPU dependencies in tests."""

    if training_size < 1:
        raise ValueError("training_size must be positive")
    steps_per_epoch = math.ceil(training_size / config.effective_batch_size)
    return {
        "output_dir": config.output_dir,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_length": config.max_length,
        "completion_only_loss": True,
        "eval_strategy": "steps",
        "eval_steps": steps_per_epoch,
        "save_strategy": "steps",
        "save_steps": steps_per_epoch,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": max(1, steps_per_epoch // 5),
        "gradient_checkpointing": True,
        "fp16": True,
        "bf16": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": "none",
    }


def save_training_result(path: str | Path, result: Mapping[str, Any]) -> None:
    """Save a reproducible result record after a run."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(result), indent=2, sort_keys=True) + "\n")


def config_record(config: TrainingConfig) -> dict[str, Any]:
    return asdict(config)
