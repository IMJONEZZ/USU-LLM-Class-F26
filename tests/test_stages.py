import json
import types

import pytest

from src import config, stages


def test_stage_times_split_the_cold_start_at_the_server_marks():
    marks = {"import_epoch": 1010.0, "launch_epoch": 1010.5, "ready_epoch": 1130.5}
    result = stages.stage_times(marks, sent_epoch=1000.0, waited_s=131.0, ttft_s=137.0)
    assert result == pytest.approx(
        {
            "container_boot": 10.0,
            "imports": 0.5,
            "vllm_startup": 120.0,
            "routing_gap": 0.5,
            "first_request": 6.0,
        }
    )
    assert sum(result.values()) == pytest.approx(137.0)


def test_stage_names_match_what_stage_times_returns():
    marks = {"import_epoch": 5.0, "launch_epoch": 6.0, "ready_epoch": 9.0}
    result = stages.stage_times(marks, sent_epoch=1.0, waited_s=9.0, ttft_s=12.0)
    assert list(result) == stages.STAGES


def test_wait_for_no_containers_polls_until_our_app_has_none(monkeypatch):
    ours = {"container_id": "ta-1", "app_name": stages.modal_server.app.name}
    other = {"container_id": "ta-2", "app_name": "some-other-app"}
    outputs = iter([json.dumps([ours, other]), json.dumps([other])])
    sleeps = []
    monkeypatch.setattr(
        stages.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(stdout=next(outputs)),
    )
    monkeypatch.setattr(stages.time, "sleep", sleeps.append)
    stages.wait_for_no_containers()
    assert sleeps == [config.RETRY_INTERVAL_SECONDS]


def test_cold_start_refuses_a_run_that_never_hit_a_503(monkeypatch):
    marks = {"import_epoch": 1010.0, "launch_epoch": 1010.5, "ready_epoch": 1130.5}
    warm = {"run": "warm", "retries": 0, "waited_s": 0.3, "ttft_s": 0.5}
    monkeypatch.setattr(stages, "wait_for_no_containers", lambda: None)
    monkeypatch.setattr(stages.time, "time", lambda: 1000.0)
    monkeypatch.setattr(stages.inference, "stream_runs", lambda q: [warm])
    monkeypatch.setattr(
        stages, "read_logs", lambda: stages.modal_server.marks_line(marks)
    )
    with pytest.raises(ValueError, match="already warm"):
        stages.cold_start("Q?")


def test_cold_start_adds_the_stages_to_the_cold_record_and_keeps_every_metric(
    monkeypatch,
):
    marks = {"import_epoch": 1010.0, "launch_epoch": 1010.5, "ready_epoch": 1130.5}
    cold = {"run": "cold start", "retries": 3, "waited_s": 131.0, "ttft_s": 137.0}
    cold["rate_tok_s"] = 13.0
    warm = {"run": "warm", "rate_tok_s": 23.0, "gate_pass": True}
    monkeypatch.setattr(stages, "wait_for_no_containers", lambda: None)
    monkeypatch.setattr(stages.time, "time", lambda: 1000.0)
    monkeypatch.setattr(stages.inference, "stream_runs", lambda q: [cold, warm])
    monkeypatch.setattr(
        stages, "read_logs", lambda: stages.modal_server.marks_line(marks)
    )
    cold_out, warm_out = stages.cold_start("Q?")
    assert cold_out["vllm_startup"] == pytest.approx(120.0)
    assert cold_out["rate_tok_s"] == 13.0
    assert "vllm_startup" not in warm_out
    assert warm_out["gate_pass"] is True


def test_find_marks_reads_the_line_the_server_prints():
    marks = {"import_epoch": 1.0, "launch_epoch": 2.0, "ready_epoch": 3.0}
    log = "noise\n" + stages.modal_server.marks_line(marks) + "\nmore noise\n"
    assert stages.find_marks(log) == marks


def test_find_marks_takes_the_latest_start_in_the_log():
    old = {"import_epoch": 1.0, "launch_epoch": 2.0, "ready_epoch": 3.0}
    new = {"import_epoch": 10.0, "launch_epoch": 20.0, "ready_epoch": 30.0}
    log = stages.modal_server.marks_line(old) + "\nx\n"
    log += stages.modal_server.marks_line(new) + "\n"
    assert stages.find_marks(log)["ready_epoch"] == 30.0


def test_find_marks_fails_loudly_when_the_log_has_none():
    with pytest.raises(ValueError):
        stages.find_marks("nothing useful here\n")


def test_cold_start_refuses_marks_from_an_earlier_start(monkeypatch):
    marks = {"import_epoch": 1.0, "launch_epoch": 2.0, "ready_epoch": 3.0}
    run = {"run": "cold start", "retries": 3, "waited_s": 10.0, "ttft_s": 12.0}
    monkeypatch.setattr(stages, "wait_for_no_containers", lambda: None)
    monkeypatch.setattr(stages.time, "time", lambda: 1000.0)
    monkeypatch.setattr(stages.inference, "stream_runs", lambda q: [run])
    monkeypatch.setattr(
        stages, "read_logs", lambda: stages.modal_server.marks_line(marks)
    )
    with pytest.raises(ValueError):
        stages.cold_start("Q?")
