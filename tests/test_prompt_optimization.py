"""A10 tests, including shared HTML checks and the isolated A9 reference. No real inference."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from dspy.utils.exceptions import AdapterParseError

from src import a9_reference
from src import a9_reference as html_generator
from src import html_evaluation as html_validate
from src import prompt_optimization as workflow
from src.a9_reference import HtmlGenerationError, HtmlGenerator, load_guidance_model
from src.html_evaluation import (
    HtmlValidationError,
    evaluate_html,
    html_grammar,
    make_html,
    metric,
    validate_html,
)
from src.prompt_optimization import report

HTML = "<html>\n<head>\n<title>Campus</title>\n</head>\n<body>\n<h1>Welcome</h1>\n<p>Our campus is open.</p>\n</body>\n</html>\n"
REQ = {"counts": {"title": 1, "h1": 1, "p": 1}, "texts": {"h1": ["Welcome"]}}


@pytest.fixture(autouse=True)
def local_test_dataset(tmp_path, monkeypatch):
    # CI creates its own inputs and never depends on a student's ignored fixtures.
    rows = [
        {
            "id": f"{split}-{i + 1:02d}",
            "split": split,
            "request": f"Create the Campus page, request {split}-{i + 1}.",
            "requirements": REQ,
        }
        for split, size in (("train", 10), ("validation", 5), ("test", 5))
        for i in range(size)
    ]
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps({"examples": rows, "config": {"challenge_id": "test-02"}})
    )
    monkeypatch.setattr(workflow, "DATASET", path)


def test_perfect_output_and_metric():
    result = evaluate_html(HTML, REQ)
    assert result["score"] == 1
    assert result["grammar_pass"] and result["tag_nesting_valid"]
    assert metric(SimpleNamespace(requirements=REQ), SimpleNamespace(html=HTML)) == 1


@pytest.mark.parametrize("document", ["", "not HTML", "```html\n" + HTML + "```"])
def test_complete_failure_and_surrounding_prose(document):
    result = evaluate_html(document, REQ)
    assert not result["grammar_pass"]
    assert not result["tag_nesting_valid"]
    if not document or document == "not HTML":
        assert result["score"] == 0


def test_partial_content_and_wrong_counts():
    result = evaluate_html(HTML.replace("Welcome", "Other"), REQ)
    assert result["grammar_pass"]
    assert result["score"] == 0.875
    missing = evaluate_html(HTML.replace("<h1>Welcome</h1>\n", ""), REQ)
    assert 0 < missing["score"] < result["score"]
    assert not missing["instruction_pass"]


@pytest.mark.parametrize(
    "document",
    [
        HTML.replace("</p>", "</h2>"),
        HTML.replace("<p>", '<p class="x">'),
        HTML + "extra",
        HTML + HTML,
        HTML.replace("<p>", "<script>").replace("</p>", "</script>"),
        HTML.replace("<h1>Welcome</h1>", "<h1></h1>"),
        "<!--comment-->" + HTML,
        "<!DOCTYPE html>" + HTML,
        "<?xml x?>" + HTML,
        HTML.replace("<p>Our campus is open.</p>", "<p/>"),
    ],
)
def test_rejects_malformed_and_empty_requirements(document):
    result = evaluate_html(document, REQ)
    assert result["score"] < 1
    assert not result["grammar_pass"]


def test_formatting_is_not_repaired():
    without_newline = HTML.rstrip("\n")
    result = evaluate_html(without_newline, REQ)
    assert not result["grammar_pass"]
    assert result["boundary_newline_match"]
    assert result["score"] == 1.0  # Transport whitespace must not block all demos.
    assert result["tag_nesting_valid"]
    assert not evaluate_html(HTML.replace("\n", ""), REQ)["grammar_pass"]


def test_dataset_has_fixed_disjoint_splits_and_no_gold_outputs():
    data = json.loads(workflow.DATASET.read_text())["examples"]
    assert len({row["id"] for row in data}) == 20
    assert [
        sum(row["split"] == split for row in data)
        for split in ("train", "validation", "test")
    ] == [10, 5, 5]
    assert all(
        row["requirements"]["counts"] and row["requirements"]["texts"] for row in data
    )
    assert all("html" not in row for row in data)


# Integration tests mock inference, while using real DSPy state serialization.


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    _, splits, config = workflow.read_inputs()
    return tmp_path, splits, config


def fake_predictor(lm, document=HTML, mode="normal"):
    def predict(**kwargs):
        if mode != "no_history":
            lm.history.append(
                {
                    "outputs": [document],
                    "kwargs": kwargs["config"],
                    "usage": {"completion_tokens": 50},
                }
            )
        lm.api_requests.append(
            {
                "options": kwargs["config"],
                "format": "json" if mode == "constrained" else None,
            }
        )
        if mode == "parse_error":
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=workflow.HtmlPage,
                lm_response="missing field",
            )
        return SimpleNamespace(html=document)

    return predict


def test_prediction_preserves_calls_outputs_and_failures():
    example = SimpleNamespace(id="x", request="page", requirements=REQ)
    for mode in ("normal", "parse_error"):
        lm = SimpleNamespace(history=[], api_requests=[])
        result = workflow.predict_once(fake_predictor(lm, mode=mode), example, lm, 11)
        assert result["seed"] == 11 and result["raw_response"] == [HTML]
        assert result["score"] == (1 if mode == "normal" else 0)
        assert bool(result["error"]) == (mode == "parse_error")
    for mode, message in [
        ("no_history", "inference call"),
        ("constrained", "Constrained"),
    ]:
        lm = SimpleNamespace(history=[], api_requests=[])
        with pytest.raises(RuntimeError, match=message):
            workflow.predict_once(fake_predictor(lm, mode=mode), example, lm, 11)
    lm = SimpleNamespace(history=[], api_requests=[])

    def missing_api(**kwargs):
        lm.history.append({"outputs": [], "kwargs": {}})
        return SimpleNamespace(html=HTML)

    with pytest.raises(RuntimeError, match="fresh Ollama"):
        workflow.predict_once(missing_api, example, lm, 11)


def test_gpu_gate_rejects_cpu_and_loads_without_prompt(monkeypatch):
    _, _, config = workflow.read_inputs()
    response = MagicMock()
    post = MagicMock(return_value=response)
    monkeypatch.setattr(workflow.requests, "post", post)
    monkeypatch.setattr(workflow.requests, "get", lambda *a, **k: response)
    loaded = {"name": config["model"], "size": 10, "size_vram": 10}
    response.json.return_value = {"models": [loaded]}
    assert workflow.require_gpu(config) == loaded
    assert "prompt" not in post.call_args.kwargs["json"]
    loaded["size_vram"] = 1
    with pytest.raises(RuntimeError, match="CPU"):
        workflow.require_gpu(config)


def test_configuration_forwards_settings_and_audits_actual_requests(monkeypatch):
    _, _, config = workflow.read_inputs()
    lm = MagicMock()
    constructor = MagicMock(return_value=lm)
    configure = MagicMock()
    monkeypatch.setattr(workflow.dspy, "LM", constructor)
    monkeypatch.setattr(workflow.dspy, "configure", configure)
    assert workflow.configure(config) is lm
    assert constructor.call_args.kwargs["cache"] is False
    assert constructor.call_args.kwargs["num_gpu"] == 99
    assert constructor.call_args.kwargs["temperature"] == 0.5
    assert configure.call_args.kwargs["adapter"].use_json_adapter_fallback is False
    lm.audit_logger.log_pre_api_call(
        "qwen",
        [],
        {"additional_args": {"complete_input_dict": {"options": {"seed": 33}}}},
    )
    assert lm.api_requests[0]["options"]["seed"] == 33


def test_optimization_split_isolation_and_state_reload(isolated, monkeypatch):
    output, splits, config = isolated
    lm = SimpleNamespace(history=[])
    monkeypatch.setattr(workflow, "require_gpu", lambda _: {"size_vram": 10})
    monkeypatch.setattr(workflow, "configure", lambda _: lm)
    optimizer = MagicMock()

    def compile_program(student, **kwargs):
        assert kwargs["trainset"] is splits["train"]
        assert kwargs["valset"] is splits["validation"]
        assert kwargs["minibatch"] is False
        assert all(set(e.inputs()) == {"request"} for e in kwargs["trainset"])
        assert not {e.id for e in splits["test"]} & {e.id for e in kwargs["trainset"]}
        student.signature = student.signature.with_instructions(
            "Optimized instructions"
        )
        return student

    optimizer.compile.side_effect = compile_program
    constructor = MagicMock(return_value=optimizer)
    monkeypatch.setattr(workflow.dspy, "MIPROv2", constructor)
    workflow.optimize(splits, config)
    assert constructor.call_args.kwargs["metric_threshold"] == 1.0
    assert constructor.call_args.kwargs["max_labeled_demos"] == 0
    baseline = workflow.program()
    baseline.load(output / "baseline.json")
    assert baseline.signature.instructions == workflow.BASELINE_INSTRUCTIONS
    restored = workflow.program()
    restored.load(output / "optimized.json")
    assert restored.signature.instructions == "Optimized instructions"
    assert "Optimized instructions" in (output / "prompts.md").read_text()


def test_smoke_success_and_failures(isolated, monkeypatch):
    output, splits, config = isolated
    monkeypatch.setattr(workflow, "require_gpu", lambda _: {})
    monkeypatch.setattr(workflow, "configure", lambda _: None)
    monkeypatch.setattr(
        workflow, "predict_once", lambda *a: {"error": None, "score": 0.8}
    )
    monkeypatch.setattr(
        workflow.dspy,
        "Predict",
        lambda *a: lambda **k: SimpleNamespace(instruction="Use HTML"),
    )
    workflow.smoke(splits, config)
    assert (output / "smoke.json").exists()
    monkeypatch.setattr(
        workflow.dspy, "Predict", lambda *a: lambda **k: SimpleNamespace(instruction="")
    )
    with pytest.raises(RuntimeError, match="empty"):
        workflow.smoke(splits, config)
    monkeypatch.setattr(workflow, "predict_once", lambda *a: {"error": "bad"})
    with pytest.raises(RuntimeError, match="parsed"):
        workflow.smoke(splits, config)


def test_evaluation_records_30_calls_and_resumes(isolated, monkeypatch):
    output, splits, config = isolated
    monkeypatch.setattr(workflow, "require_gpu", lambda _: {})
    monkeypatch.setattr(workflow, "configure", lambda _: None)
    for name in ("baseline", "optimized"):
        workflow.program().save(output / f"{name}.json")
    calls = []

    def predict(predictor, example, lm, seed):
        calls.append((example.id, seed))
        return {
            "request_id": example.id,
            "request": example.request,
            "seed": seed,
            "html": HTML,
            "raw_response": [HTML],
            **evaluate_html(HTML, example.requirements),
        }

    monkeypatch.setattr(workflow, "predict_once", predict)
    workflow.evaluate(splits, config)
    workflow.evaluate(splits, config)
    assert len(calls) == 30
    assert len((output / "responses.jsonl").read_text().splitlines()) == 30
    assert len(json.loads((output / "held_out.json").read_text())) == 5
    report(output, config)
    assert (output / "comparison.png").read_bytes().startswith(b"\x89PNG")
    assert "&lt;html&gt;" in (output / "before_after.html").read_text()
    assert "| baseline |" in (output / "results.md").read_text()
    with (output / "responses.jsonl").open("a") as stream:
        stream.write((output / "responses.jsonl").read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="30 unique"):
        report(output, config)
    with pytest.raises(ValueError, match="inputs changed"):
        workflow.evaluate(splits, {**config, "temperature": 0.7})


@pytest.mark.parametrize("command", ["smoke", "report"])
def test_cli_dispatch(isolated, monkeypatch, command):
    import sys

    from src import prompt_optimization as prompt_results

    called = MagicMock()
    monkeypatch.setattr(sys, "argv", ["a10", command])
    monkeypatch.setattr(workflow, "smoke", called)
    monkeypatch.setattr(prompt_results, "report", called)
    workflow.main()
    called.assert_called_once()


def test_list_order_is_a_requirement_and_document_order_is_structural():
    ordered = HTML.replace(
        "<p>Our campus is open.</p>", "<ol>\n<li>First</li>\n<li>Second</li>\n</ol>"
    )
    requirements = {"counts": {"ol": 1, "li": 2}, "texts": {"li": ["First", "Second"]}}
    assert evaluate_html(ordered, requirements)["score"] == 1
    reversed_items = (
        ordered.replace("First", "TEMP")
        .replace("Second", "First")
        .replace("TEMP", "Second")
    )
    assert not evaluate_html(reversed_items, requirements)["instruction_checks"][
        "list_item_order"
    ]
    wrong_order = (
        "<html><body><h1>Welcome</h1></body><head><title>Campus</title></head></html>"
    )
    assert not evaluate_html(wrong_order, REQ)["structural_checks"]["document_shape"]


VALID_HTML = """<html>
<head>
<title>Campus Guide</title>
</head>
<body>
<h1>Welcome</h1>
<p>Utah State welcomes you.</p>
<ul>
<li>Aggies</li>
</ul>
</body>
</html>
"""


def test_grammar_match_accepts_valid_and_rejects_invalid():
    grammar = html_grammar()
    assert grammar.match(VALID_HTML) is not None
    assert grammar.match("just text") is None
    assert (
        grammar.match(
            "<html><head><title>t</title></head><body><script>x</script></body></html>"
        )
        is None
    )
    assert (
        grammar.match(
            "<html><head><title>t</title></head><body><p onclick='x'>x</p></body></html>"
        )
        is None
    )


def test_validate_accepts_whitelist_document():
    assert validate_html(VALID_HTML) == VALID_HTML


@pytest.mark.parametrize(
    "document",
    [
        "",
        "just text",
        "<html><body><script>x</script></body></html>",
        "<html><body><p onclick='x'>x</p></body></html>",
        "<html><body><p>1</p>",
        "<html><body><br></br></body></html>",
        "<html><body><p>a < b</p></body></html>",
        "<div>nope</div>",
        "<html><body><script/></body></html>",
        '<html><body><br class="x"/></body></html>',
        "<html><body></script></body></html>",
        "<html><body></html></body>",
        "<body><p>x</p></body>",
        "<html><head><title>t</title></head><body><p>x</p></body></html >",
    ],
)
def test_validate_rejects_invalid_documents(document):
    with pytest.raises(HtmlValidationError):
        validate_html(document)


def test_validate_rejects_self_closing_non_void_and_br_end():
    with pytest.raises(HtmlValidationError):
        validate_html("<html><body><p/></body></html>")
    parser = html_validate._WhitelistParser()
    with pytest.raises(HtmlValidationError):
        parser.handle_endtag("br")


def test_validate_parser_exception_path():
    class Broken(html_validate._WhitelistParser):
        def feed(self, data):
            raise RuntimeError("boom")

    with (
        patch("src.html_evaluation._WhitelistParser", Broken),
        pytest.raises(HtmlValidationError, match="Malformed"),
    ):
        validate_html("<html></html>")


def test_generator_uses_guidance_path(monkeypatch):
    generator = HtmlGenerator(model_factory=lambda: MagicMock())
    monkeypatch.setattr(generator, "_generate_with_guidance", lambda prompt: VALID_HTML)
    assert generator.generate("  Tell me about USU  ") == VALID_HTML

    with pytest.raises(HtmlGenerationError):
        generator.generate("   ")


def test_generator_rejects_invalid_document(monkeypatch):
    generator = HtmlGenerator(model_factory=lambda: MagicMock())
    monkeypatch.setattr(
        generator,
        "_generate_with_guidance",
        lambda prompt: "<p>not a full document</p>",
    )
    with pytest.raises(HtmlGenerationError):
        generator.generate("hi")


def test_generator_maps_validation_failure(monkeypatch):
    generator = HtmlGenerator(model_factory=lambda: MagicMock())
    monkeypatch.setattr(generator, "_generate_with_guidance", lambda prompt: VALID_HTML)
    with (
        patch(
            "src.a9_reference.validate_html",
            side_effect=HtmlValidationError("bad"),
        ),
        pytest.raises(HtmlGenerationError, match="bad"),
    ):
        generator.generate("hi")


def test_generator_maps_backend_failure(monkeypatch):
    generator = HtmlGenerator(model_factory=lambda: MagicMock())

    def boom(_prompt: str) -> str:
        raise RuntimeError("cuda died")

    monkeypatch.setattr(generator, "_generate_with_guidance", boom)
    with pytest.raises(HtmlGenerationError, match="cuda died"):
        generator.generate("hi")


def test_generator_reraises_html_generation_error(monkeypatch):
    generator = HtmlGenerator(model_factory=lambda: MagicMock())

    def boom(_prompt: str) -> str:
        raise HtmlGenerationError("empty capture")

    monkeypatch.setattr(generator, "_generate_with_guidance", boom)
    with pytest.raises(HtmlGenerationError, match="empty capture"):
        generator.generate("hi")


def test_generate_with_guidance_empty_capture():
    class EmptyCapture:
        def __iadd__(self, other):
            return self

        def __getitem__(self, key):
            return "   "

    generator = HtmlGenerator(model_factory=lambda: EmptyCapture(), temperature=0.0)
    with (
        patch("src.a9_reference.system"),
        patch("src.a9_reference.user"),
        patch("src.a9_reference.assistant"),
        pytest.raises(HtmlGenerationError, match="empty"),
    ):
        generator._generate_with_guidance("prompt")


def test_load_guidance_model(monkeypatch):
    monkeypatch.setenv("A9_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    monkeypatch.delenv("A9_LOAD_IN_4BIT", raising=False)
    monkeypatch.setattr(html_generator.torch.cuda, "is_available", lambda: False)
    fake = MagicMock(name="transformers_model")
    with patch("src.a9_reference.Transformers", return_value=fake) as ctor:
        model = load_guidance_model()
    assert model is fake
    ctor.assert_called_once()
    assert ctor.call_args.args[0] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert ctor.call_args.kwargs["echo"] is False
    assert ctor.call_args.kwargs["dtype"] is html_generator.torch.float32
    assert ctor.call_args.kwargs["enable_backtrack"] is False
    assert ctor.call_args.kwargs["enable_ff_tokens"] is False
    assert "load_in_4bit" not in ctor.call_args.kwargs


def test_load_guidance_model_4bit(monkeypatch):
    monkeypatch.setenv("A9_LOAD_IN_4BIT", "true")
    monkeypatch.setattr(html_generator.torch.cuda, "is_available", lambda: True)
    fake = MagicMock()
    with patch("src.a9_reference.Transformers", return_value=fake) as ctor:
        load_guidance_model("Qwen/Qwen2.5-1.5B-Instruct")
    assert ctor.call_args.kwargs["quantization_config"].load_in_4bit is True
    assert ctor.call_args.kwargs["dtype"] is html_generator.torch.float16


def test_reset_guidance_engine_cache():
    engine = MagicMock()
    engine._past_key_values = object()
    engine._cached_logits = object()
    engine._cached_token_ids = [1, 2, 3]
    lm = MagicMock()
    lm.engine = engine
    html_generator._reset_guidance_engine_cache(lm)
    assert engine._past_key_values is None
    assert engine._cached_logits is None
    assert engine._cached_token_ids == []
    html_generator._reset_guidance_engine_cache(MagicMock(spec=[]))  # no engine


def test_make_html_is_guidance_callable():
    assert callable(make_html)


@pytest.mark.parametrize("document", ["<br/>", "&#32;", "&#32;<html></html>"])
def test_validator_rejects_outside_elements_and_encoded_prefix(document):
    with pytest.raises(HtmlValidationError):
        validate_html(document)


def test_guidance_capture_returns_complete_document():
    class Capture:
        def __iadd__(self, other):
            return self

        def __getitem__(self, key):
            return VALID_HTML

    generator = HtmlGenerator(model_factory=Capture)
    with (
        patch("src.a9_reference.system"),
        patch("src.a9_reference.user"),
        patch("src.a9_reference.assistant"),
    ):
        assert generator._generate_with_guidance("A page") == VALID_HTML


def test_gpu_only_loader_rejects_cpu_and_forces_cuda(monkeypatch):
    monkeypatch.setattr(html_generator.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        load_guidance_model(gpu_only=True)
    monkeypatch.setattr(html_generator.torch.cuda, "is_available", lambda: True)
    monkeypatch.delenv("A9_LOAD_IN_4BIT", raising=False)
    with patch("src.a9_reference.Transformers") as constructor:
        load_guidance_model(gpu_only=True)
    assert constructor.call_args.kwargs["device_map"] == {"": "cuda:0"}


@pytest.fixture
def reference_setup(tmp_path, monkeypatch):
    a10, output = tmp_path / "a10", tmp_path / "a9"
    a10.mkdir()
    config = workflow.read_inputs()[2]
    examples = [
        row
        for row in json.loads(workflow.DATASET.read_text())["examples"]
        if row["split"] == "test"
    ]
    (a10 / "evaluation_config.json").write_text(json.dumps(config))
    (a10 / "held_out.json").write_text(json.dumps(examples))
    monkeypatch.setattr(a9_reference.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(a9_reference.torch.cuda, "manual_seed_all", lambda _: None)
    monkeypatch.setattr(
        a9_reference.torch.cuda, "get_device_name", lambda _: "mock GPU"
    )
    monkeypatch.setattr(a9_reference.torch.cuda, "max_memory_allocated", lambda: 1000)
    monkeypatch.setattr(a9_reference.requests, "post", MagicMock())
    model = SimpleNamespace(
        parameters=lambda: [SimpleNamespace(device=SimpleNamespace(type="cuda"))],
        config=SimpleNamespace(_commit_hash="commit"),
        device="cuda:0",
    )
    loader = MagicMock(
        return_value=SimpleNamespace(engine=SimpleNamespace(model_obj=model))
    )
    monkeypatch.setattr(a9_reference, "load_guidance_model", loader)
    generator = MagicMock()
    generator._generate_with_guidance.return_value = HTML
    monkeypatch.setattr(
        a9_reference, "HtmlGenerator", MagicMock(return_value=generator)
    )
    return a10, output, loader, generator, config


def test_reference_loads_once_generates_15_and_preserves_failures(reference_setup):
    a10, output, loader, generator, config = reference_setup
    generator._generate_with_guidance.side_effect = [ValueError("bad output")] + [
        HTML
    ] * 14
    a9_reference.run_reference(a10, output)
    loader.assert_called_once_with(a9_reference.DEFAULT_MODEL_ID, gpu_only=True)
    assert generator._generate_with_guidance.call_count == 15
    rows = [
        json.loads(line)
        for line in (output / "responses.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 15 and rows[0]["score"] == 0
    assert "bad output" in rows[0]["error"]
    assert (output / "test-01-22.html").read_text() == HTML
    assert json.loads((output / "summary.json").read_text())["errors"] == 1
    with pytest.raises(FileExistsError):
        a9_reference.run_reference(a10, output)
    # Combined graph validates both experiments and refuses an incomplete reference.
    a10_rows = [
        {**row, "program": name} for name in ("baseline", "optimized") for row in rows
    ]
    (a10 / "responses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in a10_rows)
    )
    report(a10, config, reference=output)
    assert "| a9_reference |" in (a10 / "results.md").read_text()
    assert (a10 / "before_after.html").read_text().count("<section>") == 2
    (output / "responses.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="15 unique"):
        report(a10, config, reference=output)


def test_reference_rejects_cpu_before_generation(reference_setup, monkeypatch):
    a10, output, loader, generator, _ = reference_setup
    monkeypatch.setattr(a9_reference.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback"):
        a9_reference.run_reference(a10, output)
    loader.assert_not_called()
    generator._generate_with_guidance.assert_not_called()


def test_reference_rejects_partial_offload(reference_setup):
    a10, output, loader, generator, _ = reference_setup
    loader.return_value.engine.model_obj.parameters = lambda: [
        SimpleNamespace(device=SimpleNamespace(type="cpu"))
    ]
    with pytest.raises(RuntimeError, match="All model parameters"):
        a9_reference.run_reference(a10, output)
    generator._generate_with_guidance.assert_not_called()


def test_reference_does_not_fall_back_after_oom(reference_setup):
    a10, output, _, generator, _ = reference_setup
    generator._generate_with_guidance.side_effect = torch.cuda.OutOfMemoryError("OOM")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        a9_reference.run_reference(a10, output)
    assert not (output / "responses.jsonl").exists()


def test_dataset_defaults_and_missing_file(monkeypatch, tmp_path):
    bundle = json.loads(workflow.DATASET.read_text())
    bundle.pop("config")
    workflow.DATASET.write_text(json.dumps(bundle))
    _, splits, config = workflow.read_inputs()
    assert config["model"] == "qwen2.5:1.5b"
    assert config["challenge_id"] == splits["test"][0].id
    monkeypatch.setattr(workflow, "DATASET", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="--dataset"):
        workflow.read_inputs()


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_id",
        "missing_split",
        "empty_seeds",
        "duplicate_seeds",
        "wrong_challenge",
    ],
)
def test_reject_invalid_custom_dataset(problem):
    bundle = json.loads(workflow.DATASET.read_text())
    if problem == "duplicate_id":
        bundle["examples"][1]["id"] = bundle["examples"][0]["id"]
    elif problem == "missing_split":
        bundle["examples"] = [r for r in bundle["examples"] if r["split"] != "test"]
    elif problem == "empty_seeds":
        bundle["config"]["seeds"] = []
    elif problem == "duplicate_seeds":
        bundle["config"]["seeds"] = [11, 11]
    else:
        bundle["config"]["challenge_id"] = "train-01"
    workflow.DATASET.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        workflow.read_inputs()


def test_cli_accepts_custom_path_and_explains_missing_input(
    isolated, monkeypatch, capsys
):
    import sys

    custom = workflow.DATASET.with_name("custom.json")
    custom.write_bytes(workflow.DATASET.read_bytes())
    run = MagicMock()
    monkeypatch.setattr(workflow, "smoke", run)
    monkeypatch.setattr(sys, "argv", ["a10", "smoke", "--dataset", str(custom)])
    workflow.main()
    run.assert_called_once()
    assert workflow.DATASET == custom
    monkeypatch.setattr(
        sys, "argv", ["a10", "smoke", "--dataset", str(custom.with_name("absent.json"))]
    )
    with pytest.raises(SystemExit) as error:
        workflow.main()
    assert error.value.code == 2
    assert "Dataset not found" in capsys.readouterr().err


def test_report_uses_custom_dataset_sizes_and_one_seed(tmp_path):
    _, splits, config = workflow.read_inputs()
    example = dict(splits["test"][0])
    config = {**config, "challenge_id": example["id"], "seeds": [99]}
    row = {
        "request_id": example["id"],
        "request": example["request"],
        "seed": 99,
        "html": HTML,
        "raw_response": [HTML],
        **evaluate_html(HTML, REQ),
    }
    workflow.write_json(tmp_path / "held_out.json", [example])
    (tmp_path / "responses.jsonl").write_text(
        "".join(
            json.dumps({**row, "program": name}) + "\n"
            for name in ["baseline", "optimized"]
        )
    )
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "responses.jsonl").write_text(
        json.dumps({**row, "program": "a9_reference"}) + "\n"
    )
    report(tmp_path, config, reference=reference)
    text = (tmp_path / "results.md").read_text()
    assert "1/1 (100.0%)" in text
    assert "1 held-out requests, 1 fresh generations" in text
    assert (tmp_path / "before_after.html").read_text().count("<section>") == 2


def test_instruction_proposal_retries_parse_failure_without_json_fallback(
    monkeypatch, caplog
):
    signature = workflow.dspy.Signature("task -> proposed_instruction")
    error = AdapterParseError(
        adapter_name="ChatAdapter",
        signature=signature,
        lm_response="echoed input fields",
    )
    request = MagicMock(
        side_effect=[error, [{"proposed_instruction": "Return HTML only."}]]
    )
    monkeypatch.setattr(workflow.dspy.ChatAdapter, "__call__", request)
    adapter = workflow.ProposalRetryAdapter(use_json_adapter_fallback=False)
    assert adapter(None, {}, signature, [], {}) == [
        {"proposed_instruction": "Return HTML only."}
    ]
    assert request.call_count == 2
    assert "retrying (attempt 2/3)" in caplog.text
    assert adapter.use_json_adapter_fallback is False


@pytest.mark.parametrize("proposal, expected_calls", [(True, 3), (False, 1)])
def test_proposal_retries_are_bounded_and_html_is_never_retried(
    monkeypatch, proposal, expected_calls
):
    signature = (
        workflow.dspy.Signature("task -> proposed_instruction")
        if proposal
        else workflow.HtmlPage
    )
    error = AdapterParseError(
        adapter_name="ChatAdapter", signature=signature, lm_response="missing output"
    )
    request = MagicMock(side_effect=error)
    monkeypatch.setattr(workflow.dspy.ChatAdapter, "__call__", request)
    adapter = workflow.ProposalRetryAdapter(use_json_adapter_fallback=False)
    with pytest.raises(AdapterParseError):
        adapter(None, {}, signature, [], {})
    assert request.call_count == expected_calls


def test_proposal_retry_does_not_hide_infrastructure_failure(monkeypatch):
    request = MagicMock(side_effect=ConnectionError("Ollama unavailable"))
    monkeypatch.setattr(workflow.dspy.ChatAdapter, "__call__", request)
    adapter = workflow.ProposalRetryAdapter(use_json_adapter_fallback=False)
    with pytest.raises(ConnectionError):
        adapter(
            None, {}, workflow.dspy.Signature("task -> proposed_instruction"), [], {}
        )
    assert request.call_count == 1
