"""CPU-only checks for A6 conversation preparation and split integrity."""

import json
import re
from pathlib import Path

import pytest

from src.trainer import (
    TrainingConfig,
    contains_assignment_question,
    deduplicate_conversations,
    format_conversation,
    grouped_train_development_split,
    load_own_examples,
    near_assignment_question,
    prepare_candidate_pool,
    sample_across_sources,
    sample_with_followup_quota,
    save_training_result,
    training_arguments,
    validate_conversation,
)


def chat(question, answer, *followups):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    for followup, response in followups:
        messages.extend(
            [
                {"role": "user", "content": followup},
                {"role": "assistant", "content": response},
            ]
        )
    return messages


def row(row_id, question, answer="answer", *followups):
    return {"id": row_id, "messages": chat(question, answer, *followups)}


def test_format_preserves_followup_context_and_trains_final_answer():
    messages = chat("Initial question?", "First answer.", ("Why?", "Because."))

    formatted = format_conversation(messages)

    assert [message["role"] for message in formatted["prompt"]] == [
        "user",
        "assistant",
        "user",
    ]
    assert formatted["prompt"][-1]["content"] == "Why?"
    assert formatted["completion"] == [{"role": "assistant", "content": "Because."}]


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "Question?"}],
        chat("Question?", "  "),
        [
            {"role": "assistant", "content": "Answer."},
            {"role": "user", "content": "Question?"},
        ],
    ],
)
def test_incomplete_or_invalid_conversation_is_rejected(messages):
    with pytest.raises(ValueError):
        validate_conversation(messages)


def test_exact_assignment_question_is_rejected_even_in_followup():
    forbidden = "How many Ls are in the word parallel?"
    messages = chat("Count letters in a word.", "Sure.", (forbidden, "Three."))

    assert contains_assignment_question(messages)
    with pytest.raises(ValueError, match="assignment question"):
        format_conversation(messages)


def test_deduplicate_whole_chats_but_not_shared_generic_followups():
    original = row(1, "How does a lamp work?", "Light.", ("Continue", "More."))
    punctuation_variant = row(2, "How does a lamp work", "Light", ("continue", "More"))
    different_context = row(
        3, "How does a bicycle work?", "Pedals.", ("Continue", "More.")
    )

    unique = deduplicate_conversations(
        [original, punctuation_variant, different_context]
    )

    assert [item["id"] for item in unique] == [1, 3]


def test_candidate_pool_filters_near_copy_bad_structure_and_duplicate():
    rows = [
        {"source": "A", "messages": chat("How does rain form?", "Condensation.")},
        {"source": "A", "messages": chat("How does rain form", "Condensation")},
        {
            "source": "B",
            "messages": chat(
                "If there is a shark in the basement, is it safe to go upstairs?",
                "Yes.",
            ),
        },
        {"source": "B", "messages": chat("Incomplete?", "Answer", ("And then?", ""))},
        {
            "source": "C",
            "messages": chat("Explain a bicycle.", "Pedals.", ("Why?", "Gears.")),
        },
    ]

    assert near_assignment_question(rows[2]["messages"])
    selected = prepare_candidate_pool(rows, excluded_ids=frozenset())

    assert [candidate["id"] for candidate in selected] == [0, 4]
    assert [candidate["format"] for candidate in selected] == ["single", "followup"]


def test_source_sample_is_reproducible_and_spreads_sources():
    rows = [{"id": i, "source": "A" if i < 20 else "B"} for i in range(25)]

    sample = sample_across_sources(rows, 8)

    assert sample == sample_across_sources(rows, 8)
    assert len(sample) == 8
    assert {item["source"] for item in sample} == {"A", "B"}


def test_locked_followups_are_used_and_missing_ones_raise():
    rows = [{"id": i, "source": "A", "format": "single"} for i in range(30)]
    rows += [{"id": 100 + i, "source": "B", "format": "followup"} for i in range(6)]
    approved = frozenset({100, 101, 102})

    sample = sample_with_followup_quota(rows, 12, 3, approved_followup_ids=approved)

    assert {r["id"] for r in sample if r["format"] == "followup"} == approved
    with pytest.raises(ValueError):
        sample_with_followup_quota(
            rows, 12, 3, approved_followup_ids=frozenset({100, 999, 102})
        )


def test_excluded_sources_and_terse_instructions_are_filtered():
    rows = [
        {"source": "Dove", "messages": chat("Explain tides.", "Moon pull.")},
        {"source": "GOAT", "messages": chat("What is 2+2?", "4")},
        {"source": "TheoremQA", "messages": chat("Prove a lemma.", "Proof.")},
        {
            "source": "TaskSource",
            "messages": chat("With no explanation, label this: great.", "positive"),
        },
        {
            "source": "TaskSource",
            "messages": chat("Pixels: #112233#445566#778899", "x"),
        },
    ]

    selected = prepare_candidate_pool(rows, excluded_ids=frozenset())

    assert [candidate["source"] for candidate in selected] == ["Dove"]


def test_followup_quota_is_exact_and_reproducible():
    rows = [{"id": i, "source": "A", "format": "single"} for i in range(30)]
    rows += [{"id": 100 + i, "source": "B", "format": "followup"} for i in range(10)]

    sample = sample_with_followup_quota(rows, 12, 3)

    assert len(sample) == 12
    assert sum(item["format"] == "followup" for item in sample) == 3
    assert sample == sample_with_followup_quota(rows, 12, 3)


def test_grouped_split_keeps_repeated_first_questions_together():
    rows = [
        row(0, "What is the same question?", "A"),
        row(1, "What is the same question?", "B", ("Why?", "C")),
    ] + [row(index, f"Unique question {index}?") for index in range(2, 30)]

    train, dev = grouped_train_development_split(rows, development_fraction=0.1)
    train_ids = {item["id"] for item in train}
    dev_ids = {item["id"] for item in dev}

    assert train_ids.isdisjoint(dev_ids)
    assert train_ids | dev_ids == set(range(30))
    assert {0, 1} <= train_ids or {0, 1} <= dev_ids
    assert (train, dev) == grouped_train_development_split(
        rows, development_fraction=0.1
    )


def test_training_arguments_follow_controlled_recipe_and_completion_only_loss():
    config = TrainingConfig(max_length=512, output_dir="a6-test")

    arguments = training_arguments(config, training_size=500)

    assert config.effective_batch_size == 8
    assert arguments["eval_strategy"] == "epoch"
    assert arguments["save_strategy"] == "epoch"
    assert arguments["max_length"] == 512
    assert arguments["completion_only_loss"] is True
    assert arguments["load_best_model_at_end"] is True


def test_save_training_result_creates_readable_json(tmp_path):
    path = tmp_path / "run" / "result.json"

    save_training_result(path, {"loss": 0.5, "examples": 500})

    assert json.loads(path.read_text()) == {"examples": 500, "loss": 0.5}


OWN_EXAMPLES = Path(__file__).parents[1] / "assignment_6" / "own_examples.jsonl"
ASSIGNMENT_TOPICS = re.compile(
    r"sphinx|feather|shark|knuckle|wood\s?chuck|talos|parallel|\bopera\b"
    r"|british pound|president|christmas|basement|boxes under|four legs",
    re.IGNORECASE,
)


def test_own_examples_are_valid_and_avoid_assignment_topics():
    rows = load_own_examples(OWN_EXAMPLES)
    questions = [row["messages"][0]["content"] for row in rows]

    assert len(rows) >= 100
    assert len(set(questions)) == len(questions)
    assert not [r for r in rows if ASSIGNMENT_TOPICS.search(json.dumps(r["messages"]))]


def test_load_own_examples_rejects_an_assignment_question(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps({"messages": chat("Was Talos alive?", "No.")}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_own_examples(path)


TARGETED = Path(__file__).parents[1] / "assignment_6" / "targeted_examples.jsonl"


def test_targeted_examples_cover_every_question_and_avoid_assignment_wording():
    rows = [
        json.loads(line) for line in TARGETED.read_text(encoding="utf-8").splitlines()
    ]
    by_tag = {}
    for row in rows:
        by_tag.setdefault(row["tag"], []).append(row)
    questions = [row["messages"][0]["content"] for row in rows]

    assert sorted(by_tag, key=lambda t: int(t[1:])) == [f"Q{n}" for n in range(1, 11)]
    # Q2 intentionally prunes repeated equal-mass and unrelated conversions.
    assert len(by_tag["Q2"]) == 32
    assert all(len(group) >= 40 for tag, group in by_tag.items() if tag != "Q2")
    assert len(set(questions)) == len(questions)
    assert not any(contains_assignment_question(r["messages"]) for r in rows)
    assert not any(near_assignment_question(r["messages"]) for r in rows)
    assert len(load_own_examples(TARGETED)) == len(rows)


def test_q6_pruning_keeps_numeric_supply_limits():
    rows = [
        json.loads(line) for line in TARGETED.read_text(encoding="utf-8").splitlines()
    ]
    q6 = [row for row in rows if row["tag"] == "Q6"]
    excluded_concepts = (
        "songs",
        "shelve",
        "textbooks",
        "loaves of bread",
        "test tubes",
        "sails",
        "memory cards",
        "toddler",
        "worm",
        "ant carry",
        "sparrow",
        "bee visit",
        "cat eat",
        "dog bury",
        "child eat",
        "snail",
        "hen produce",
        "bicycle travel",
        "fish read",
        "goldfish climb",
        "calculator make",
        "rock eat",
    )
    for row in q6:
        question, answer = [message["content"] for message in row["messages"]]
        assert not any(concept in question.lower() for concept in excluded_concepts)
        assert re.search(r"\d", answer)
        assert not answer.startswith(("None.", "Nothing.", "The supply isn't"))
    # Retain both direct quantity bounds and explicit resource conversions.
    assert any(
        "No more than 40 kilograms" in row["messages"][1]["content"] for row in q6
    )
    assert any(
        "100 grams divided by 2 grams" in row["messages"][1]["content"] for row in q6
    )
