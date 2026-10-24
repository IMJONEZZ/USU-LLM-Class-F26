from src import config, modal_server
from tests.helpers import value_after


def test_vllm_command_takes_names_and_length_from_config():
    command = modal_server.vllm_command()
    assert value_after(command, "--served-model-name") == config.SERVED_BASE_NAME
    assert value_after(command, "--max-model-len") == str(config.EVAL_MAX_SEQ_LENGTH)
    assert "--enable-lora" not in command


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
