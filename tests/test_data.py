import pytest

from src import config, data

# the 24 kept IFEval types and their prompt counts, measured from the real dataset; they sum to 300
REAL_COUNTS = [19, 19, 18, 17, 17, 16, 14, 13, 13, 13, 13, 13]
REAL_COUNTS += [12, 11, 11, 11, 10, 10, 10, 9, 8, 8, 8, 7]


def fake_examples(counts):
    examples, key = [], 1000
    for type_number, count in enumerate(counts):
        for _ in range(count):
            examples.append(
                {
                    "key": key,
                    "prompt": f"p{key}",
                    "instruction_id": f"type{type_number}",
                    "kwargs": {},
                }
            )
            key += 1
    return examples


def test_to_example_drops_none_kwargs_and_keeps_the_real_ones():
    none_row = {
        "key": 1001,
        "prompt": "No commas please",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{"num_highlights": None, "relation": None}],
    }
    highlight_row = {
        "key": 1002,
        "prompt": "Highlight 3 sections",
        "instruction_id_list": ["detectable_format:number_highlighted_sections"],
        "kwargs": [{"num_highlights": 3, "relation": None}],
    }
    assert data.to_example(none_row) == {
        "key": 1001,
        "prompt": "No commas please",
        "instruction_id": "punctuation:no_comma",
        "kwargs": {},
    }
    assert data.to_example(highlight_row)["kwargs"] == {"num_highlights": 3}


def test_to_example_skips_prompts_with_more_than_one_instruction():
    row = {
        "key": 1,
        "prompt": "two things",
        "instruction_id_list": ["a:b", "c:d"],
        "kwargs": [{}, {}],
    }
    assert data.to_example(row) is None


def test_types_at_the_cutoff_are_dropped_and_one_above_is_kept():
    examples = fake_examples([6, 7])
    kept, dropped = data.drop_small_types(examples, min_prompts=6)
    assert dropped == {"type0": 6}
    assert {e["instruction_id"] for e in kept} == {"type1"}
    assert len(kept) == 7


@pytest.mark.parametrize("seed", range(5))
def test_every_real_sized_type_reaches_all_three_splits(seed):
    examples = fake_examples(REAL_COUNTS)
    splits = data.split_examples(examples, seed)
    all_keys = splits["train"] + splits["validation"] + splits["test"]
    assert sorted(all_keys) == sorted(
        e["key"] for e in examples
    )  # no overlap, none lost
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [
        135,
        90,
        75,
    ]
    for counts in data.split_counts(examples, splits).values():
        assert all(count >= 1 for count in counts.values())


def test_same_seed_gives_the_same_split_and_another_seed_differs():
    examples = fake_examples(REAL_COUNTS)
    assert data.split_examples(examples, config.SEED) == data.split_examples(
        examples, config.SEED
    )
    assert data.split_examples(examples, 1) != data.split_examples(examples, 2)


def test_split_counts_tallies_each_type_in_each_split():
    examples = fake_examples([2, 1])  # keys 1000, 1001 are type0; 1002 is type1
    splits = {"train": [1000, 1002], "validation": [1001], "test": []}
    assert data.split_counts(examples, splits) == {
        "type0": {"train": 1, "validation": 1, "test": 0},
        "type1": {"train": 1, "validation": 0, "test": 0},
    }


def test_load_examples_returns_the_rows_and_only_the_single_instruction_tasks(
    monkeypatch,
):
    rows = [
        {
            "key": 1,
            "prompt": "a",
            "instruction_id_list": ["x:y"],
            "kwargs": [{"n": None, "m": 2}],
        },
        {
            "key": 2,
            "prompt": "b",
            "instruction_id_list": ["x:y", "z:w"],
            "kwargs": [{}, {}],
        },
    ]
    monkeypatch.setattr("datasets.load_dataset", lambda name, split: rows)
    all_rows, tasks = data.load_examples()
    assert all_rows == rows
    assert tasks == [
        {"key": 1, "prompt": "a", "instruction_id": "x:y", "kwargs": {"m": 2}}
    ]


def test_main_writes_the_split_once_and_afterwards_only_compares(
    tmp_path, monkeypatch, capsys
):
    examples = fake_examples(REAL_COUNTS)
    path = tmp_path / "assets" / "splits.json"
    monkeypatch.setattr(data, "load_examples", lambda: (examples, examples))
    monkeypatch.setattr(config, "SPLITS_PATH", str(path))
    data.main()
    assert f"wrote {path}" in capsys.readouterr().out
    saved = path.read_text()
    data.main()
    assert "matches this run's split: True" in capsys.readouterr().out
    assert path.read_text() == saved
