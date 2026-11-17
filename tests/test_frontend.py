"""Tests for the frontend's prompt, history, generation, and UI outputs."""

import importlib
import sys
from contextlib import nullcontext
from datetime import date
from types import ModuleType
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def frontend_module():
    """Import the frontend without requiring Modal or starting a deployment.

    The no-op decorators preserve the function for local tests, while the fake
    image builder lets the module-level container configuration be evaluated.
    """

    modal_stub = ModuleType("modal")

    class FakeApp:
        """Provide only the Modal app methods used while importing frontend.py."""

        def __init__(self, name):
            self.name = name

        def function(self, **kwargs):
            return lambda function: function

    class FakeImage:
        """Support the image-building chain without installing container packages."""

        @staticmethod
        def debian_slim():
            return FakeImage()

        def pip_install(self, *packages):
            return self

    modal_stub.App = FakeApp
    modal_stub.Image = FakeImage
    modal_stub.concurrent = lambda **kwargs: lambda function: function
    modal_stub.asgi_app = lambda: lambda function: function

    with patch.dict(sys.modules, {"modal": modal_stub}):
        sys.modules.pop("src.frontend", None)
        return importlib.import_module("src.frontend")


def test_build_system_prompts_includes_game_date_and_score(frontend_module):
    """Verify prompt text contains the selected game, date, and both scores."""
    prompts = frontend_module.build_system_prompts(
        [("Away at Home", (17, 24))], current_date=date(2026, 11, 17)
    )

    assert list(prompts) == ["Away at Home"]
    assert "Today's date is 2026-11-17" in prompts["Away at Home"]
    assert "Away team: 17" in prompts["Away at Home"]
    assert "Home team: 24" in prompts["Away at Home"]
    assert frontend_module.load_history(None, "Away at Home") == []


def test_process_message_returns_reply_and_keeps_context(frontend_module):
    """Check visible replies and confirm prior turns reach the reply generator."""
    game = "Away at Home"
    histories = {
        game: [
            {"role": "user", "content": "Earlier update"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "system", "content": "Ignore unsupported saved turns"},
            {"role": "user", "content": 42},
        ],
        "Other game": [{"role": "user", "content": "Keep this thread"}],
    }
    prompts = {game: "Game system instructions"}
    model_inputs = []

    def fake_generate(messages):
        model_inputs.append(messages)
        return "Updated score: 24-17"

    visible, textbox, updated_histories = frontend_module.process_message(
        "Add injury context", histories, game, prompts, fake_generate
    )

    assert visible == [
        {"role": "user", "content": "Earlier update"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Add injury context"},
        {"role": "assistant", "content": "Updated score: 24-17"},
    ]
    assert textbox == ""
    assert updated_histories["Other game"] == [
        {"role": "user", "content": "Keep this thread"}
    ]
    assert model_inputs == [
        [
            {"role": "system", "content": "Game system instructions"},
            {"role": "user", "content": "Earlier update"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Add injury context"},
        ]
    ]

    visible, textbox, updated_histories = frontend_module.process_message(
        "Another detail", updated_histories, game, prompts, fake_generate
    )
    assert visible[-2:] == [
        {"role": "user", "content": "Another detail"},
        {"role": "assistant", "content": "Updated score: 24-17"},
    ]
    assert model_inputs[-1] == [
        {"role": "system", "content": "Game system instructions"},
        {"role": "user", "content": "Earlier update"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Add injury context"},
        {"role": "assistant", "content": "Updated score: 24-17"},
        {"role": "user", "content": "Another detail"},
    ]
    assert textbox == ""
    assert len(updated_histories[game]) == 6


def test_new_hypothesis_resets_only_the_selected_game(frontend_module):
    """Check reset starts a new selected-game thread and preserves other games."""
    first_game = "Away at Home"
    second_game = "Visitors at Hosts"
    histories = {
        first_game: [{"role": "user", "content": "Old hypothesis"}],
        second_game: [{"role": "user", "content": "Keep this one"}],
    }
    captured_inputs = []

    visible, textbox, updated_histories = frontend_module.process_message(
        "Try a different assumption",
        histories,
        first_game,
        {first_game: "First game prompt"},
        lambda messages: captured_inputs.append(messages) or "New hypothesis reply",
        reset_history=True,
    )

    assert visible == [
        {"role": "user", "content": "Try a different assumption"},
        {"role": "assistant", "content": "New hypothesis reply"},
    ]
    assert captured_inputs == [
        [
            {"role": "system", "content": "First game prompt"},
            {"role": "user", "content": "Try a different assumption"},
        ]
    ]
    assert updated_histories[second_game] == [
        {"role": "user", "content": "Keep this one"}
    ]
    assert textbox == ""


def test_empty_reset_clears_thread_without_generating(frontend_module):
    """Check blank submissions skip generation and blank resets clear history."""
    game = "Away at Home"
    histories = {game: [{"role": "user", "content": "Prior turn"}]}

    visible, textbox, updated_histories = frontend_module.process_message(
        "  ", histories, game, {game: "Prompt"}, lambda messages: pytest.fail()
    )
    assert visible == histories[game]

    visible, textbox, updated_histories = frontend_module.process_message(
        "",
        updated_histories,
        game,
        {game: "Prompt"},
        lambda messages: pytest.fail(),
        True,
    )
    assert visible == []
    assert textbox == ""
    assert updated_histories[game] == []

    # Unexpected state values recover to an empty per-game history dictionary.
    visible, textbox, empty_histories = frontend_module.process_message(
        "", None, game, {game: "Prompt"}, lambda messages: pytest.fail()
    )
    assert visible == []
    assert textbox == ""
    assert empty_histories == {game: []}


def test_generation_error_is_returned_as_chat_output(frontend_module):
    """Verify generation failures appear as assistant text and are saved."""
    game = "Away at Home"
    visible, textbox, histories = frontend_module.process_message(
        "Update this prediction",
        {},
        game,
        {game: "Prompt"},
        lambda messages: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    assert visible[-1] == {
        "role": "assistant",
        "content": "Generation failed (RuntimeError): model unavailable",
    }
    assert histories[game] == visible
    assert textbox == ""


def test_generate_answer_loads_model_and_decodes_output(frontend_module):
    """Exercise model loading, chat formatting, generation, and decoded output.

    Fake Torch and Transformers modules keep the test deterministic and avoid
    downloading weights or requiring a GPU.
    """
    loaded = {"tokenizers": 0, "models": 0, "generated": 0}

    class FakeInputIds:
        shape = (1, 1)

    class FakeInputs(dict):
        input_ids = FakeInputIds()

        def to(self, device):
            assert device == "fake-device"
            return self

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1]["content"] == "A question"
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "formatted prompt"

        def __call__(self, prompts, **kwargs):
            assert prompts == ["formatted prompt"]
            assert kwargs == {"return_tensors": "pt"}
            return FakeInputs(input_ids=self.input_ids)

        input_ids = FakeInputIds()

        def decode(self, tokens, **kwargs):
            assert tokens == ["generated tokens"]
            assert kwargs == {"skip_special_tokens": True}
            return "A decoded model answer"

    tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "test/model"
            loaded["tokenizers"] += 1
            return tokenizer

    class FakeModel:
        device = "fake-device"

        def eval(self):
            self.evaluated = True

        def generate(self, **kwargs):
            assert "input_ids" in kwargs
            assert kwargs["max_new_tokens"] == 512
            loaded["generated"] += 1
            return [["prompt token", "generated tokens"]]

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            assert model_id == "test/model"
            assert kwargs["torch_dtype"] == "fake-fp16"
            assert kwargs["device_map"] == "auto"
            loaded["models"] += 1
            return FakeModel()

    torch_stub = ModuleType("torch")
    torch_stub.float16 = "fake-fp16"
    torch_stub.inference_mode = nullcontext
    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoTokenizer = FakeAutoTokenizer
    transformers_stub.AutoModelForCausalLM = FakeAutoModel

    state = {"model": None, "tokenizer": None}
    with patch.dict(
        sys.modules,
        {"torch": torch_stub, "transformers": transformers_stub},
    ):
        answer = frontend_module.generate_answer(
            [{"role": "user", "content": "A question"}],
            "test/model",
            state,
            frontend_module.Lock(),
        )
        second_answer = frontend_module.generate_answer(
            [{"role": "user", "content": "A question"}],
            "test/model",
            state,
            frontend_module.Lock(),
        )

    assert answer == "A decoded model answer"
    assert second_answer == "A decoded model answer"
    assert loaded == {"tokenizers": 1, "models": 1, "generated": 2}


def test_gradio_callbacks_return_chat_and_restore_game_threads(
    frontend_module, monkeypatch
):
    """Exercise submit, game-switch, and reset callbacks through their outputs."""

    class FakeContext:
        """Stand in for Gradio layout containers used as context managers."""

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class FakeBlocks(FakeContext):
        """Capture event handlers registered while the Gradio block is built."""

        current = None

        def __init__(self):
            self.events = {}
            FakeBlocks.current = self

    class FakeComponent:
        """Record callback bindings so the test can invoke the real handlers."""

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def submit(self, function, **kwargs):
            FakeBlocks.current.events["submit"] = function

        def click(self, function, **kwargs):
            FakeBlocks.current.events["click"] = function

        def change(self, function, **kwargs):
            FakeBlocks.current.events["change"] = function

    gradio_stub = ModuleType("gradio")
    gradio_stub.Blocks = FakeBlocks
    gradio_stub.Markdown = FakeComponent
    gradio_stub.State = lambda value: value
    gradio_stub.Row = FakeContext
    gradio_stub.Column = FakeContext
    gradio_stub.Radio = FakeComponent
    gradio_stub.Chatbot = FakeComponent
    gradio_stub.Textbox = FakeComponent
    gradio_stub.Button = FakeComponent
    gradio_routes_stub = ModuleType("gradio.routes")
    gradio_routes_stub.mount_gradio_app = lambda api, blocks, path: blocks
    fastapi_stub = ModuleType("fastapi")
    fastapi_stub.FastAPI = object
    monkeypatch.setitem(sys.modules, "gradio", gradio_stub)
    monkeypatch.setitem(sys.modules, "gradio.routes", gradio_routes_stub)
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_stub)

    generated_prompts = []

    def fake_generate(messages, model_id, model_state, model_lock):
        generated_prompts.append(messages)
        return f"Answer for {messages[-1]['content']}"

    monkeypatch.setattr(frontend_module, "generate_answer", fake_generate)
    app = frontend_module.frontend()
    first_game = "Oregon at Michigan St. (11/20)"
    second_game = "Iowa at Illinois (11/21)"

    first_result = app.events["submit"]("First question", {}, first_game)
    saved_threads = first_result[2]
    assert first_result[0][-1] == {
        "role": "assistant",
        "content": "Answer for First question",
    }
    assert app.events["change"](saved_threads, second_game) == []

    second_result = app.events["submit"]("Second question", saved_threads, second_game)
    saved_threads = second_result[2]
    assert app.events["change"](saved_threads, first_game) == first_result[0]

    reset_result = app.events["click"]("New question", saved_threads, first_game)
    assert reset_result[0] == [
        {"role": "user", "content": "New question"},
        {"role": "assistant", "content": "Answer for New question"},
    ]
    assert reset_result[2][second_game] == second_result[0]
    assert generated_prompts[-1][1:] == [{"role": "user", "content": "New question"}]
