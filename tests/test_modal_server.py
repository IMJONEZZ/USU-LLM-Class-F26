import pytest

from src import config, modal_server
from tests.helpers import value_after


def test_vllm_command_takes_rank_names_and_length_from_config():
    command = modal_server.vllm_command(fast_boot=True)
    assert value_after(command, "--max-lora-rank") == str(config.LORA_R)
    assert value_after(command, "--served-model-name") == config.SERVED_BASE_NAME
    assert value_after(command, "--lora-modules") == (
        f"{config.ADAPTER_NAME}={config.ADAPTER_PATH}"
    )
    assert value_after(command, "--max-model-len") == str(config.EVAL_MAX_SEQ_LENGTH)


def test_fast_boot_switches_eager_mode_on_and_off():
    assert "--enforce-eager" in modal_server.vllm_command(fast_boot=True)
    assert "--enforce-eager" not in modal_server.vllm_command(fast_boot=False)


def test_adapter_path_sits_under_a_mounted_volume():
    mount = config.ADAPTER_PATH.rsplit("/", 1)[0]
    assert mount in modal_server.volumes


def test_wait_until_ready_polls_health_until_it_answers(monkeypatch):
    answers = iter([OSError("refused"), OSError("503"), None])
    sleeps = []

    def fake_urlopen(url, timeout):
        answer = next(answers)
        if answer is not None:
            raise answer

    monkeypatch.setattr(modal_server.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(modal_server.time, "sleep", sleeps.append)
    modal_server.wait_until_ready()
    assert len(sleeps) == 2


def test_marks_record_the_three_epochs_and_the_current_fast_boot_setting(monkeypatch):
    monkeypatch.setattr(modal_server, "FAST_BOOT", False)
    marks = modal_server.build_marks(launch_epoch=5.0, ready_epoch=9.0)
    assert marks == {
        "import_epoch": modal_server.IMPORT_EPOCH,
        "launch_epoch": 5.0,
        "ready_epoch": 9.0,
        "fast_boot": False,
    }


@pytest.mark.parametrize(
    ("environ", "expected"),
    [({}, True), ({"FAST_BOOT": "1"}, True), ({"FAST_BOOT": "0"}, False)],
    ids=["unset means on", "one means on", "zero means off"],
)
def test_fast_boot_setting_is_read_from_the_environment(environ, expected):
    assert modal_server.fast_boot_setting(environ) is expected
