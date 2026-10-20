import json

import pytest

from src import chart, config


def _record(**changes):
    record = {
        "container_boot": 5.0,
        "imports": 0.5,
        "vllm_startup": 130.0,
        "routing_gap": 10.0,
        "first_request": 6.0,
        "ttft_s": 151.5,
        "vllm_download_s": None,
        "vllm_model_load_s": 3.0,
        "vllm_torch_compile_s": None,
        "vllm_graph_capture_s": None,
        "vllm_init_engine_s": 80.0,
        "fast_boot": True,
        "hf_cache_files": 7,
        "vllm_cache_files": 0,
    }
    record.update(changes)
    return record


def test_segments_split_vllm_startup_without_overlap_and_sum_to_the_total():
    parts = chart.segments(_record())
    assert list(parts) == chart.SEGMENTS
    assert parts["weights load"] == 3.0
    assert parts["profile + warmup"] == 80.0
    assert parts["other vLLM"] == 47.0
    assert sum(parts.values()) == pytest.approx(151.5)


def test_compile_time_comes_out_of_the_engine_init_block():
    parts = chart.segments(_record(vllm_torch_compile_s=20.0, vllm_graph_capture_s=5.0))
    assert parts["compile + graph capture"] == 25.0
    assert parts["profile + warmup"] == 55.0
    assert sum(parts.values()) == pytest.approx(151.5)


def test_download_time_comes_out_of_the_weights_block():
    parts = chart.segments(_record(vllm_download_s=12.0, vllm_model_load_s=15.0))
    assert parts["download"] == 12.0
    assert parts["weights load"] == 3.0
    assert sum(parts.values()) == pytest.approx(151.5)


def test_steps_vllm_did_not_print_fall_into_other_vllm_and_are_not_guessed():
    parts = chart.segments(_record(vllm_model_load_s=None, vllm_init_engine_s=None))
    assert parts["other vLLM"] == 130.0
    assert parts["weights load"] == 0
    assert sum(parts.values()) == pytest.approx(151.5)


def test_run_label_names_the_run_and_its_recorded_state():
    label = chart.run_label(_record(fast_boot=False), "run 3")
    assert label == "run 3\nFAST_BOOT off · HF cache 7 files · vLLM cache 0 files"


def test_print_table_lists_every_segment_with_its_seconds_and_the_total(capsys):
    chart.print_table([_record()], ["run 1"])
    rows = {}
    for line in capsys.readouterr().out.splitlines():
        rows[line.rsplit(None, 1)[0].strip()] = line.split()[-1]
    assert rows["profile + warmup"] == "80.0"
    assert rows["other vLLM"] == "47.0"
    assert rows["total"] == "151.5"
    assert all(label in rows for label in chart.SEGMENTS)


def test_label_text_is_dark_on_light_fills_and_white_on_dark_fills():
    assert chart.text_on("#eda100") == chart.INK
    assert chart.text_on("#6250d6") == "white"


def test_main_draws_the_chart_png_from_the_saved_runs(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(config, "RUNS_DIR", str(tmp_path))
    for name in ("run_a", "run_b"):
        saved = [_record(), {"run": "warm"}]
        (tmp_path / f"{name}.json").write_text(json.dumps(saved))
    chart.main(["run_a", "run_b"])
    assert (tmp_path / "m8_chart.png").stat().st_size > 5000
    assert "run_a" in capsys.readouterr().out
