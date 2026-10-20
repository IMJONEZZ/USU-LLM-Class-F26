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
    monkeypatch.setattr(stages, "wait_for_no_containers", lambda: None)
    monkeypatch.setattr(
        stages.inference, "stream", lambda q, m: {"retries": 0, "waited_s": 0.3}
    )
    with pytest.raises(ValueError):
        stages.cold_start("Q?")
