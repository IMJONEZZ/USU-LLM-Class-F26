import re

from src import data_gen


def test_generate_data_returns_requested_number_of_samples() -> None:
    assert data_gen.generate_data(0) == []

    samples = data_gen.generate_data(5)

    assert len(samples) == 5
    assert all(set(sample) == {"in", "out"} for sample in samples)


def test_generated_samples_resolve_markers_and_avoid_trailing_commas() -> None:
    for sample in data_gen.generate_data(20):
        for sentence in sample.values():
            assert re.search(r"<[A-Z_*][^<>]*>", sentence) is None
            assert re.search(r",\s*(?:[.!?]|\||$)", sentence) is None
        assert sample["in"][0].isupper()


def test_repeated_and_bold_verb_markers_keep_the_same_value(monkeypatch) -> None:
    monkeypatch.setattr(data_gen.random, "randrange", lambda _length: 12)
    monkeypatch.setattr(data_gen.random, "choice", lambda choices: choices[0])

    sample = data_gen.generate_data(1)[0]

    assert sample["in"].count("activate") == 2
    assert "**ACTIVATE**" in sample["out"]
    assert "<VERB" not in sample["in"] + sample["out"]


def test_distinct_numbered_verb_markers_choose_distinct_verbs(monkeypatch) -> None:
    monkeypatch.setattr(data_gen.random, "randrange", lambda _length: 0)
    monkeypatch.setattr(data_gen.random, "choice", lambda choices: choices[0])

    sample = data_gen.generate_data(1)[0]

    bold_verbs = re.findall(r"\*\*([A-Z ]+)\*\*", sample["out"])
    assert len(bold_verbs) == 2
    assert bold_verbs[0] != bold_verbs[1]
