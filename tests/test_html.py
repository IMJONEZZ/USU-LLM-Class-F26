import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

from src import html


def test_get_teams_extracts_ranked_lines():
    article = """1. Alpha Aces
Previous ranking: 2
2. Beta Bears
 3. Indented team
Not a ranking
"""

    assert html.get_teams(article) == ["Alpha Aces", "Beta Bears"]


def test_run_model_uses_guidance_with_modal_data(monkeypatch, capsys):
    app_data = {
        "teams": ["Alpha Aces", "Beta Bears"],
        "text": "A short article about two teams.",
    }
    monkeypatch.setattr(html, "app_data", app_data)

    calls = {"select": [], "transformers": None, "captured": None}

    class FakeLanguageModel:
        def __iadd__(self, value):
            return self

        def __add__(self, value):
            return self

        def __getitem__(self, key):
            assert key == "html_text"
            return "<html>mock report</html>"

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name():
            return "Mock T4"

    def guidance_decorator(**kwargs):
        def decorate(function):
            def invoke(*args, **kwargs):
                return function(FakeLanguageModel(), *args, **kwargs)

            return invoke

        return decorate

    def select(*, options):
        calls["select"].append(options)
        return options[0]

    def capture(value, *, name):
        calls["captured"] = name
        return value

    def with_temperature(value, *, temperature):
        assert temperature == 0.7
        return value

    guidance_module = ModuleType("guidance")
    guidance_module.guidance = guidance_decorator
    guidance_module.system = nullcontext
    guidance_module.user = nullcontext
    guidance_module.assistant = nullcontext
    guidance_module.gen = lambda **kwargs: "generated text"
    guidance_module.select = select

    chat_module = ModuleType("guidance.chat")
    chat_module.Qwen2dot5ChatTemplate = object()

    models_module = ModuleType("guidance.models")

    def transformers(model_id, **kwargs):
        calls["transformers"] = (model_id, kwargs)
        return FakeLanguageModel()

    models_module.Transformers = transformers

    library_module = ModuleType("guidance.library")
    library_module.one_or_more = lambda value: value
    library_module.capture = capture
    library_module.with_temperature = with_temperature

    monkeypatch.setitem(sys.modules, "guidance", guidance_module)
    monkeypatch.setitem(sys.modules, "guidance.chat", chat_module)
    monkeypatch.setitem(sys.modules, "guidance.models", models_module)
    monkeypatch.setitem(sys.modules, "guidance.library", library_module)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda()))
    html.run_model.local()

    out = capsys.readouterr().out
    assert "Loaded 2 teams" in out
    assert "Mock T4" in out
    assert "<html>mock report</html>" in out
    assert calls["transformers"][0] == "Qwen/Qwen2.5-14B-Instruct"
    assert calls["transformers"][1]["device_map"] == "cuda"
    assert calls["captured"] == "html_text"
    assert any(options == app_data["teams"] for options in calls["select"])
