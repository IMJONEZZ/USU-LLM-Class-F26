import types

import pytest

from src import config, experiments


def _fake_everything(monkeypatch, file_counts):
    """Record every step in order; file_counts feeds volume_file_count in call order."""
    calls = []
    counts = iter(file_counts)
    stages = experiments.stages
    monkeypatch.setattr(stages, "wait_for_no_containers", lambda: calls.append("wait"))
    monkeypatch.setattr(stages, "volume_file_count", lambda name: next(counts))
    monkeypatch.setattr(
        stages, "run_and_save", lambda q, run: calls.append(f"run {run}")
    )
    monkeypatch.setattr(experiments, "wipe_volume", lambda n: calls.append(f"wipe {n}"))
    monkeypatch.setattr(
        experiments,
        "deploy",
        lambda fast_boot: calls.append(f"deploy fast_boot={fast_boot}"),
    )
    monkeypatch.setattr(experiments.chart, "main", lambda names: calls.append("chart"))
    return calls


def test_the_five_runs_happen_in_order_with_one_change_between_neighbours(monkeypatch):
    calls = _fake_everything(monkeypatch, [0, 0, 0, 5, 5])
    experiments.run_experiments("Q?")
    assert calls == [
        "wait",
        f"wipe {config.HF_CACHE_VOLUME}",
        f"wipe {config.VLLM_CACHE_VOLUME}",
        "deploy fast_boot=True",
        "run m8_run1",
        "run m8_run2",
        "deploy fast_boot=False",
        "run m8_run3",
        "run m8_run4",
        "deploy fast_boot=True",
        "run m8_run5",
        "deploy fast_boot=True",
        "chart",
    ]


def test_run_five_is_refused_if_the_vllm_cache_is_empty_by_then(monkeypatch):
    calls = _fake_everything(monkeypatch, [0, 0, 0, 5, 0])
    with pytest.raises(ValueError, match="populated"):
        experiments.run_experiments("Q?")
    assert "run m8_run5" not in calls
    assert "chart" not in calls
    assert calls[-1] == "deploy fast_boot=True"


def test_run_four_is_refused_if_run_three_left_the_vllm_cache_empty(monkeypatch):
    calls = _fake_everything(monkeypatch, [0, 0, 0, 0])
    with pytest.raises(ValueError, match="populated"):
        experiments.run_experiments("Q?")
    assert "run m8_run4" not in calls
    assert "chart" not in calls
    assert calls[-1] == "deploy fast_boot=True"


def test_run_three_is_refused_if_the_vllm_cache_is_not_empty(monkeypatch):
    calls = _fake_everything(monkeypatch, [0, 0, 3])
    with pytest.raises(ValueError, match="empty"):
        experiments.run_experiments("Q?")
    assert "run m8_run3" not in calls
    assert calls[-1] == "deploy fast_boot=True"


def test_the_normal_configuration_is_redeployed_even_when_a_run_fails(monkeypatch):
    calls = _fake_everything(monkeypatch, [0, 0, 0, 5])

    def fail(question, run):
        raise RuntimeError("boom")

    monkeypatch.setattr(experiments.stages, "run_and_save", fail)
    with pytest.raises(RuntimeError):
        experiments.run_experiments("Q?")
    assert calls[-1] == "deploy fast_boot=True"


@pytest.mark.parametrize(("fast_boot", "expected"), [(True, "1"), (False, "0")])
def test_deploy_passes_the_fast_boot_setting_through_the_environment(
    monkeypatch, fast_boot, expected
):
    seen = {}

    def fake_run(command, env, check):
        seen["command"], seen["env"] = command, env

    monkeypatch.setattr(experiments.subprocess, "run", fake_run)
    experiments.deploy(fast_boot)
    assert seen["env"]["FAST_BOOT"] == expected
    assert "src.modal_server" in seen["command"]


def test_wipe_volume_removes_every_top_level_entry_recursively(monkeypatch):
    removed = []
    entries = [types.SimpleNamespace(path="hub"), types.SimpleNamespace(path="xet")]
    volume = types.SimpleNamespace(
        listdir=lambda path: entries,
        remove_file=lambda path, recursive: removed.append((path, recursive)),
    )
    monkeypatch.setattr(experiments.modal.Volume, "from_name", lambda name: volume)
    experiments.wipe_volume("any-volume")
    assert removed == [("hub", True), ("xet", True)]


def _saved_runs(monkeypatch, tmp_path, names):
    monkeypatch.setattr(config, "RUNS_DIR", str(tmp_path))
    for name in names:
        (tmp_path / f"{name}.json").write_text("[]")


def test_run_five_alone_runs_only_run_five_and_redraws_the_chart(monkeypatch, tmp_path):
    _saved_runs(monkeypatch, tmp_path, experiments.RUNS[:4])
    calls = _fake_everything(monkeypatch, [5, 5])
    experiments.run_five("Q?")
    assert calls == ["deploy fast_boot=True", "run m8_run5", "chart"]


def test_run_five_alone_is_refused_if_the_model_cache_is_empty(monkeypatch, tmp_path):
    _saved_runs(monkeypatch, tmp_path, experiments.RUNS[:4])
    calls = _fake_everything(monkeypatch, [0])
    with pytest.raises(ValueError, match="populated"):
        experiments.run_five("Q?")
    assert calls == []


def test_run_five_alone_is_refused_if_the_compile_cache_is_empty(monkeypatch, tmp_path):
    _saved_runs(monkeypatch, tmp_path, experiments.RUNS[:4])
    calls = _fake_everything(monkeypatch, [5, 0])
    with pytest.raises(ValueError, match="populated"):
        experiments.run_five("Q?")
    assert "run m8_run5" not in calls
    assert "chart" not in calls


def test_run_five_alone_is_refused_if_an_earlier_saved_run_is_missing(
    monkeypatch, tmp_path
):
    _saved_runs(monkeypatch, tmp_path, ["m8_run1", "m8_run2", "m8_run4"])
    calls = _fake_everything(monkeypatch, [])
    with pytest.raises(ValueError, match="m8_run3"):
        experiments.run_five("Q?")
    assert calls == []


def test_main_rejects_an_unknown_command_before_touching_anything(monkeypatch):
    calls = _fake_everything(monkeypatch, [])
    with pytest.raises(ValueError, match="unknown"):
        experiments.main("everything", "Q?")
    assert calls == []
