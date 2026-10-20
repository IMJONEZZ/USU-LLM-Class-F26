import importlib
import sys
from types import ModuleType


def load_deploy_module(monkeypatch):
    """Import deploy.py with lightweight Modal and HF SDK stand-ins."""
    calls = {}

    class FakeImage:
        def __init__(self):
            self.env_values = None

        @classmethod
        def from_registry(cls, *args, **kwargs):
            calls["image_registry"] = (args, kwargs)
            return cls()

        def entrypoint(self, value):
            calls["entrypoint"] = value
            return self

        def uv_pip_install(self, *packages):
            calls["packages"] = packages
            return self

        def env(self, values):
            self.env_values = values
            calls["image_env"] = values
            return self

    class FakeApp:
        def __init__(self, name):
            calls["app_name"] = name

        def server(self, **kwargs):
            calls["server_options"] = kwargs
            return lambda cls: cls

    class FakeVolume:
        @staticmethod
        def from_name(name, *, create_if_missing):
            calls.setdefault("volumes", []).append((name, create_if_missing))
            return f"volume:{name}"

    class FakeSecret:
        @staticmethod
        def from_local_environ(names):
            calls["secret_names"] = names
            return "fake-hf-secret"

    fake_modal = ModuleType("modal")
    fake_modal.Image = FakeImage
    fake_modal.App = FakeApp
    fake_modal.Volume = FakeVolume
    fake_modal.Secret = FakeSecret
    fake_modal.enter = lambda: lambda fn: fn
    fake_modal.exit = lambda: lambda fn: fn

    fake_huggingface_hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls["snapshot_download"] = kwargs
        return "/cached/adapter"

    fake_huggingface_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)
    monkeypatch.delitem(sys.modules, "src.deploy", raising=False)

    module = importlib.import_module("src.deploy")
    return module, calls


def test_server_startup_downloads_adapter_and_starts_vllm(monkeypatch, capsys):
    deploy, calls = load_deploy_module(monkeypatch)
    popen_calls = []

    class FakeProcess:
        def terminate(self):
            calls["terminated"] = True

    def fake_popen(command):
        popen_calls.append(command)
        return FakeProcess()

    monkeypatch.setattr(deploy.subprocess, "Popen", fake_popen)
    server = deploy.Server()
    server.startup()

    assert calls["snapshot_download"] == {
        "repo_id": deploy.ADAPTER_NAME,
        "token": True,
    }
    command = popen_calls[0]
    assert command[:4] == [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        deploy.BASE_MODEL_NAME,
    ]
    assert command[command.index("--lora-modules") + 1] == (
        f"{deploy.SERVED_MODEL_NAME}=/cached/adapter"
    )
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[-3:] == [
        "--enforce-eager",
        "--tensor-parallel-size",
        "1",
    ]
    assert "--host" in command
    assert "--port" in command
    assert "['vllm', 'serve'" in capsys.readouterr().out


def test_server_stop_terminates_process(monkeypatch):
    deploy, _ = load_deploy_module(monkeypatch)
    server = deploy.Server()

    class FakeProcess:
        terminated = False

        def terminate(self):
            self.terminated = True

    server.process = FakeProcess()
    server.stop()

    assert server.process.terminated


def test_modal_configuration_includes_t4_lora_secret_and_sampler_fallback(monkeypatch):
    deploy, calls = load_deploy_module(monkeypatch)

    assert calls["app_name"] == "example-vllm-inference"
    assert calls["image_env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert calls["secret_names"] == ["HF_TOKEN"]
    assert calls["server_options"]["gpu"] == "T4:1"
    assert calls["server_options"]["secrets"] == ["fake-hf-secret"]
    assert deploy.BASE_MODEL_NAME == "meta-llama/Llama-3.2-1B"
