import logging
import sys
from types import SimpleNamespace

from src.open_model_gpu import run_model as modal_run_model


def test_run_model_loads_quantized_model_and_generates_text(monkeypatch, capsys):
    calls = {}
    tokenizer = object()
    model = SimpleNamespace(device="cuda:0", dtype="float16")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            calls["tokenizer_model_id"] = model_id
            return tokenizer

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls["model"] = (model_id, kwargs)
            return model

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            calls["quantization"] = kwargs
            calls["quantization_config"] = self

    def pipeline(task, **kwargs):
        calls["pipeline"] = (task, kwargs)

        def generate(prompt, **generation_kwargs):
            calls["generation"] = (prompt, generation_kwargs)
            return [{"generated_text": "Once upon a time, a test."}]

        return generate

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda: "Test GPU",
        memory_allocated=lambda: 2 * 1024**3,
        memory_reserved=lambda: 3 * 1024**3,
    )
    fake_torch = SimpleNamespace(cuda=cuda)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=AutoTokenizer,
        AutoModelForCausalLM=AutoModelForCausalLM,
        BitsAndBytesConfig=BitsAndBytesConfig,
        pipeline=pipeline,
    )
    fake_subprocess = SimpleNamespace(
        run=lambda *args, **kwargs: SimpleNamespace(stdout="GPU status")
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)

    modal_run_model.local()

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    assert calls["tokenizer_model_id"] == model_id
    assert calls["quantization"] == {"load_in_8bit": True}
    assert calls["model"] == (
        model_id,
        {
            "device_map": 0,
            "quantization_config": calls["quantization_config"],
        },
    )
    assert calls["pipeline"] == (
        "text-generation",
        {"model": model, "tokenizer": tokenizer},
    )
    assert calls["generation"] == (
        "Once upon a time",
        {"max_new_tokens": 50, "do_sample": True, "temperature": 0.7},
    )
    assert capsys.readouterr().out.splitlines() == [
        "CUDA available: True",
        "GPU: Test GPU",
        "GPU status",
        "Model device: cuda:0",
        "GPU memory allocated: 2.0 GB",
        "GPU memory reserved: 3.0 GB",
        "Model dtype: float16",
        "Once upon a time, a test.",
    ]


def test_run_model_suppresses_bitsandbytes_logging(monkeypatch):
    logger = logging.getLogger("bitsandbytes")
    original_level = logger.level
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_name=lambda: "Test GPU",
                memory_allocated=lambda: 0,
                memory_reserved=lambda: 0,
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: object()),
            AutoModelForCausalLM=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(
                    device="cuda:0", dtype="float16"
                )
            ),
            BitsAndBytesConfig=lambda **kwargs: object(),
            pipeline=lambda *args, **kwargs: (
                lambda *a, **kw: [{"generated_text": "generated"}]
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "subprocess",
        SimpleNamespace(run=lambda *args, **kwargs: SimpleNamespace(stdout="")),
    )

    try:
        modal_run_model.local()
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(original_level)
