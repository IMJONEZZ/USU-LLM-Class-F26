import json
from unittest.mock import patch

from src.dataloader import create_dataloader_v1, create_dataloader_v2
from src.main import (
    load_star_wars_lines,
    load_star_wars_text,
    time_baseline_loader,
    time_v2_loader,
)


def test_load_star_wars_text_joins_lines_with_newline(tmp_path):
    data = [
        {"Character": "A", "Line": "Hello there."},
        {"Character": "B", "Line": "General Kenobi."},
    ]
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(data))

    text = load_star_wars_text(fixture_path)

    assert text == "Hello there.\nGeneral Kenobi."


def test_time_baseline_loader_returns_positive_float():
    text = "The quick brown fox jumps over the lazy dog " * 20

    duration = time_baseline_loader(
        text, batch_size=1, max_length=4, stride=2, shuffle=False, drop_last=True
    )

    assert isinstance(duration, float)
    assert duration > 0


def test_time_baseline_loader_calls_dataloader_creation_repeats_times():
    text = "The quick brown fox jumps over the lazy dog " * 20

    with patch(
        "src.main.create_dataloader_v1", wraps=create_dataloader_v1
    ) as mock_create:
        time_baseline_loader(
            text,
            repeats=3,
            batch_size=1,
            max_length=4,
            stride=2,
            shuffle=False,
            drop_last=True,
        )

    assert mock_create.call_count == 3


def test_load_star_wars_lines_returns_list_of_line_strings(tmp_path):
    data = [
        {"Character": "A", "Line": "Hello there."},
        {"Character": "B", "Line": "General Kenobi."},
    ]
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_text(json.dumps(data))

    lines = load_star_wars_lines(fixture_path)

    assert lines == ["Hello there.", "General Kenobi."]


def test_time_v2_loader_calls_dataloader_creation_repeats_times():
    lines = ["The quick brown fox jumps over the lazy dog"] * 50

    with patch(
        "src.main.create_dataloader_v2", wraps=create_dataloader_v2
    ) as mock_create:
        time_v2_loader(lines, repeats=3)

    assert mock_create.call_count == 3
