"""Tests for Star Wars dialogue grouping and language-model datasets."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.dataloader import combine_consecutive_lines, create_datasets


def test_combine_consecutive_lines_groups_only_matching_speaker_and_film():
    rows = [
        {"film": "ep4_a_new_hope", "speaker": "Leia", "text": "Help me."},
        {"film": "ep4_a_new_hope", "speaker": "Leia", "text": "You're my only hope."},
        {"film": "ep4_a_new_hope", "speaker": "Luke", "text": "Who is she?"},
        {"film": "ep5_empire_strikes_back", "speaker": "Luke", "text": "I am ready."},
    ]

    assert combine_consecutive_lines(rows) == [
        {
            "film": "ep4_a_new_hope",
            "speaker": "Leia",
            "text": "Help me. You're my only hope.",
        },
        {"film": "ep4_a_new_hope", "speaker": "Luke", "text": "Who is she?"},
        {"film": "ep5_empire_strikes_back", "speaker": "Luke", "text": "I am ready."},
    ]


def test_combine_consecutive_lines_does_not_merge_unknown_speakers():
    rows = [
        {"film": "ep4", "speaker": None, "text": "First."},
        {"film": "ep4", "speaker": None, "text": "Second."},
    ]

    assert [row["text"] for row in combine_consecutive_lines(rows)] == [
        "First.",
        "Second.",
    ]


def test_create_datasets_uses_episodes_one_through_four_and_reserves_five_six(
    monkeypatch,
):
    source = [
        {
            "film": "ep1_the_phantom_menace",
            "speaker": "Qui-Gon",
            "text": "Episode one training text.",
        },
        {
            "film": "ep4_a_new_hope",
            "speaker": "Leia",
            "text": "Episode four training text.",
        },
        {
            "film": "rogue_one",
            "speaker": "Jyn",
            "text": "Not in the training range.",
        },
        {
            "film": "ep5_empire_strikes_back",
            "speaker": "Luke",
            "text": "Validation text here.",
        },
        {
            "film": "ep6_return_of_the_jedi",
            "speaker": "Luke",
            "text": "Test text here.",
        },
    ]
    dataset_type = MagicMock()
    dataset_type.from_dict.side_effect = lambda data: data["input_ids"]
    datasets_module = SimpleNamespace(
        Dataset=dataset_type,
        load_dataset=MagicMock(return_value=source),
    )
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    tokenizer.encode.side_effect = lambda text, add_special_tokens: [
        ord(char) for char in text
    ]

    result = create_datasets(tokenizer, block_size=512)

    assert set(result) == {"train", "validation", "test"}
    assert len(result["train"]) == 2
    assert len(result["validation"]) == 1
    assert len(result["test"]) == 1
    datasets_module.load_dataset.assert_called_once_with(
        "IMJONEZZ/star-wars-dataset", "cues", split="train"
    )
    tokenized_texts = [call.args[0] for call in tokenizer.encode.call_args_list]
    assert tokenized_texts == [
        "Episode one training text.",
        "Episode four training text.",
        "Validation text here.",
        "Test text here.",
    ]


@pytest.mark.parametrize("block_size", [0, 1])
def test_create_datasets_requires_at_least_two_tokens_per_block(block_size):
    with pytest.raises(ValueError, match="at least 2"):
        create_datasets(MagicMock(), block_size=block_size)
