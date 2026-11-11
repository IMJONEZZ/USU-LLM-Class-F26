"""Unit tests for the prompt optimizer without Modal or GPU access."""

from __future__ import annotations

import importlib
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


class FakeModalDict(dict):
    """In-memory replacement for the shared Modal dictionary."""

    @classmethod
    def from_name(cls, *_args, **_kwargs):
        return cls()


class FakeModalApp:
    def __init__(self, *_args, **_kwargs):
        pass

    def local_entrypoint(self):
        return lambda function: function

    def function(self, **_kwargs):
        return lambda function: function


class FakeModalImage:
    @classmethod
    def debian_slim(cls):
        return cls()

    def pip_install(self, *_args, **_kwargs):
        return self


@pytest.fixture(scope="module")
def optimizer_module():
    """Import the source with a Modal shim so importing never contacts Modal."""
    modal_stub = types.ModuleType("modal")
    modal_stub.App = FakeModalApp
    modal_stub.Dict = FakeModalDict
    modal_stub.Image = FakeModalImage

    previous_modal = sys.modules.get("modal")
    sys.modules["modal"] = modal_stub
    try:
        return importlib.import_module("src.prompt_optimizer")
    finally:
        if previous_modal is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = previous_modal


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(self.values))

    def __getitem__(self, key):
        row, selection = key
        assert row == 0
        return self.values[selection]


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.eos_token_id = 0
        self.calls = []

    @classmethod
    def from_pretrained(cls, _model_id):
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return FakeBatch(input_ids=FakeTensor([1, 2, 3, 4]))

    def decode(self, _tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return "excellent"


class FakeModel:
    def __init__(self):
        self.device = "fake-device"
        self.generation_calls = []
        self.was_set_to_eval = False

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def eval(self):
        self.was_set_to_eval = True

    def generate(self, *, input_ids, **kwargs):
        self.generation_calls.append(kwargs)
        return FakeTensor(input_ids.values + [99])


class FakeExample:
    def __init__(self, **values):
        self.__dict__.update(values)
        self._input_names = ()

    def with_inputs(self, *names):
        self._input_names = names
        return self

    def inputs(self):
        return {name: getattr(self, name) for name in self._input_names}


class FakePrediction:
    def __init__(self, performance):
        self.performance = performance


class FakePredictor:
    def __init__(self):
        self.demos = []


class FakeChainOfThought:
    def __init__(self, _signature):
        self.predict = FakePredictor()

    def __call__(self, *, summary):
        messages = []
        for demo in self.predict.demos:
            messages.extend(
                [
                    _fake_message("user", demo.summary),
                    _fake_message("assistant", demo.performance),
                ]
            )
        messages.append(_fake_message("user", summary))
        request = SimpleNamespace(
            model="Qwen/Qwen3.5-4B",
            system="Classify the team's performance.",
            messages=messages,
            config=SimpleNamespace(max_tokens=256, temperature=0.0),
        )
        lm = _FAKE_DSPY.lm
        response = lm.engine.complete(request)
        return FakePrediction(response.message.text)


def _fake_message(role, text):
    return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])


class FakeModule:
    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def predictors(self):
        return [self.classifier.predict]


class FakeLM:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.engine = RecordingEngine(kwargs["engine"])

    def inspect_history(self, n=1):
        assert n == 1
        request = self.engine.requests[-1]
        print(request.system)
        for message in request.messages:
            print("".join(part.text for part in message.parts))


class RecordingEngine:
    def __init__(self, engine):
        self.engine = engine
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return self.engine.complete(request)


class FakeMessage:
    def __init__(self, text):
        self.text = text

    @classmethod
    def assistant(cls, text):
        return cls(text)


class FakeResponse:
    def __init__(self, *, id, model, message, finish_reason, usage):
        self.id = id
        self.model = model
        self.message = message
        self.finish_reason = finish_reason
        self.usage = usage


class FakeUsage:
    def __init__(self, *, input_tokens, output_tokens, total_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens


class FakeBootstrapFewShot:
    def __init__(self, *, metric, max_bootstrapped_demos, max_labeled_demos):
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos

    def compile(self, student, *, trainset):
        assert self.max_bootstrapped_demos == 3
        student.classifier.predict.demos = trainset[: self.max_labeled_demos]
        return student


class FakeEvaluate:
    def __init__(self, *, devset, metric, **_kwargs):
        self.devset = devset
        self.metric = metric

    def __call__(self, program):
        scores = [
            self.metric(example, program(**example.inputs()), None)
            for example in self.devset
        ]
        return sum(scores) / len(scores)


_FAKE_DSPY = None


def _install_runtime_stubs(monkeypatch):
    global _FAKE_DSPY

    dspy_stub = types.ModuleType("dspy")
    dspy_stub.__path__ = []
    dspy_stub.Module = FakeModule
    dspy_stub.Signature = type("Signature", (), {})
    dspy_stub.InputField = lambda **_kwargs: None
    dspy_stub.OutputField = lambda **_kwargs: None
    dspy_stub.Example = FakeExample
    dspy_stub.ChainOfThought = FakeChainOfThought
    dspy_stub.LM = FakeLM

    def configure(*, lm):
        dspy_stub.lm = lm

    dspy_stub.configure = configure
    _FAKE_DSPY = dspy_stub

    evaluate_stub = types.ModuleType("dspy.evaluate")
    evaluate_stub.Evaluate = FakeEvaluate
    lm15_stub = types.ModuleType("dspy.lm15")
    lm15_stub.Message = FakeMessage
    lm15_stub.Response = FakeResponse
    lm15_stub.Usage = FakeUsage
    teleprompt_stub = types.ModuleType("dspy.teleprompt")
    teleprompt_stub.BootstrapFewShot = FakeBootstrapFewShot

    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.inference_mode = nullcontext
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = FakeModel
    transformers_stub.AutoTokenizer = FakeTokenizer

    for name, module in {
        "dspy": dspy_stub,
        "dspy.evaluate": evaluate_stub,
        "dspy.lm15": lm15_stub,
        "dspy.teleprompt": teleprompt_stub,
        "torch": torch_stub,
        "transformers": transformers_stub,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_main_stores_datasets_and_starts_remote_function(optimizer_module, monkeypatch):
    calls = []
    optimizer_module.app_data.clear()
    monkeypatch.setattr(
        optimizer_module,
        "run_model",
        SimpleNamespace(remote=lambda: calls.append("started")),
    )

    optimizer_module.main()

    assert optimizer_module.app_data["train"] == optimizer_module.train_text
    assert optimizer_module.app_data["test"] == optimizer_module.test_text
    assert calls == ["started"]


def test_remote_pipeline_prints_prompts_and_uses_few_shot_demos(
    optimizer_module, monkeypatch, capsys
):
    _install_runtime_stubs(monkeypatch)
    optimizer_module.app_data.clear()
    optimizer_module.app_data["train"] = [
        ("positive review one", "excellent"),
        ("positive review two", "excellent"),
        ("positive review three", "excellent"),
    ]
    optimizer_module.app_data["test"] = [
        ("held-out positive review", "excellent"),
        ("held-out negative review", "poor"),
    ]

    optimizer_module.run_model()

    output = capsys.readouterr().out
    assert "PROMPT BEFORE OPTIMIZATION" in output
    assert "PROMPT AFTER OPTIMIZATION" in output
    assert "Few-shot examples in optimized prompt: 3" in output
    assert "held-out positive review" in output
    assert "positive review one" in output
    assert "0.5" in output

    lm = _FAKE_DSPY.lm
    engine = lm.engine
    assert len(engine.requests) >= 4
    assert len(engine.requests[0].messages) == 1
    assert len(engine.requests[1].messages) == 7
    assert engine.requests[1].messages[-1].parts[0].text == "held-out positive review"

    response = engine.complete(
        SimpleNamespace(
            model=lm.model,
            system=None,
            messages=[_fake_message("user", "sampling test")],
            config=SimpleNamespace(max_tokens=7, temperature=0.4),
        )
    )
    assert response.message.text == "excellent"
    assert response.usage.input_tokens == 4
    assert response.usage.output_tokens == 1
    assert response.usage.total_tokens == 5
    assert engine.requests[-1].messages[0].parts[0].text == "sampling test"
