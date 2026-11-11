import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from dspy.lm15 import Config, Message, Request

from src import prompt_optimization
from src.prompt_optimization import (
    TransformersEngine,
    exact_match,
    format_comparison,
    format_program_prompt,
)


def test_exact_match_ignores_non_digit_characters():
    gold = SimpleNamespace(answer="The answer is 12,345.")
    prediction = SimpleNamespace(answer="12 345")

    assert exact_match(gold, prediction)


def test_exact_match_returns_true_for_equal_digits():
    gold = SimpleNamespace(answer="0042")
    prediction = SimpleNamespace(answer="42")

    assert exact_match(gold, prediction)


def test_exact_match_returns_false_for_different_digits():
    gold = SimpleNamespace(answer="12345")
    prediction = SimpleNamespace(answer="12346")

    assert not exact_match(gold, prediction)


def test_exact_match_returns_false_when_answers_have_no_digits():
    gold = SimpleNamespace(answer="No numeric answer")
    prediction = SimpleNamespace(answer="No numeric answer")

    assert not exact_match(gold, prediction)


class FakeBatch(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    eos_token_id = 0

    def __init__(self):
        self.messages = None
        self.decoded_tokens = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        assert tokenize is False
        assert add_generation_prompt is True
        return "rendered prompt"

    def __call__(self, prompt, return_tensors):
        assert prompt == "rendered prompt"
        assert return_tensors == "pt"
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))

    def decode(self, token_ids, skip_special_tokens):
        self.decoded_tokens = token_ids.tolist()
        assert skip_special_tokens is True
        return " 42 "


class FakeModel:
    device = "cpu"

    def __init__(self, output_tokens):
        self.output_tokens = output_tokens
        self.generation_options = None

    def generate(self, input_ids, **generation_options):
        self.generation_options = generation_options
        output_ids = torch.tensor([self.output_tokens])
        return torch.cat((input_ids, output_ids), dim=1)


@pytest.fixture
def fake_engine():
    tokenizer = FakeTokenizer()
    model = FakeModel([4, 5])
    return TransformersEngine(model, tokenizer), model, tokenizer


def make_request(*, system=None, messages=(), config=None):
    if not messages:
        messages = (Message.user("Test prompt"),)
    return Request(
        model="test/model",
        messages=tuple(messages),
        system=system,
        config=config or Config(max_tokens=8, temperature=0),
    )


def test_engine_sends_system_and_user_prompts_to_tokenizer(fake_engine):
    engine, _, tokenizer = fake_engine
    request = make_request(
        system="System prompt",
        messages=(Message.user("User prompt"),),
    )

    engine.complete(request)

    assert tokenizer.messages == [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
    ]


def test_engine_maps_developer_and_system_prompts_correctly(fake_engine):
    engine, _, tokenizer = fake_engine
    request = make_request(
        system="System prompt",
        messages=(
            Message.developer("Developer prompt"),
            Message.user("User prompt"),
        ),
    )

    engine.complete(request)

    assert tokenizer.messages == [
        {"role": "system", "content": "System prompt"},
        {"role": "system", "content": "Developer prompt"},
        {"role": "user", "content": "User prompt"},
    ]


def test_engine_passes_max_output_tokens_to_model(fake_engine):
    engine, model, _ = fake_engine

    engine.complete(make_request(config=Config(max_tokens=13, temperature=0)))

    assert model.generation_options["max_new_tokens"] == 13


def test_engine_samples_when_temperature_is_positive(fake_engine):
    engine, model, _ = fake_engine

    engine.complete(
        make_request(config=Config(max_tokens=8, temperature=0.7, top_p=0.9))
    )

    assert model.generation_options["do_sample"] is True
    assert model.generation_options["temperature"] == 0.7
    assert model.generation_options["top_p"] == 0.9


def test_engine_uses_greedy_generation_at_zero_temperature(fake_engine):
    engine, model, _ = fake_engine

    engine.complete(make_request(config=Config(max_tokens=8, temperature=0)))

    assert model.generation_options["do_sample"] is False
    assert "temperature" not in model.generation_options


def test_engine_decodes_only_generated_tokens(fake_engine):
    engine, _, tokenizer = fake_engine

    prediction = engine.complete(make_request())

    assert tokenizer.decoded_tokens == [4, 5]
    assert prediction.message.text == "42"


def test_engine_reports_generated_token_usage_and_finish_reason(fake_engine):
    engine, _, _ = fake_engine

    prediction = engine.complete(
        make_request(config=Config(max_tokens=3, temperature=0))
    )

    assert prediction.usage.input_tokens == 3
    assert prediction.usage.output_tokens == 2
    assert prediction.usage.output_tokens <= 3
    assert prediction.usage.total_tokens == 5
    assert prediction.finish_reason == "stop"


def test_engine_marks_output_as_truncated_at_token_limit():
    model = FakeModel([4, 5, 6])
    engine = TransformersEngine(model, FakeTokenizer())

    prediction = engine.complete(
        make_request(config=Config(max_tokens=3, temperature=0))
    )

    assert prediction.usage.output_tokens == 3
    assert prediction.finish_reason == "length"


def test_format_program_prompt_includes_program_instructions():
    program = SimpleNamespace(
        predictors=lambda: [
            SimpleNamespace(
                signature=SimpleNamespace(instructions="Answer arithmetic questions."),
                demos=[],
            )
        ]
    )

    formatted_prompt = format_program_prompt(program)

    assert "Instructions: Answer arithmetic questions." in formatted_prompt


def test_format_program_prompt_works_without_demonstrations():
    program = SimpleNamespace(
        predictors=lambda: [
            SimpleNamespace(
                signature=SimpleNamespace(instructions="Answer carefully."),
                demos=[],
            )
        ]
    )

    formatted_prompt = format_program_prompt(program)

    assert "Instructions: Answer carefully." in formatted_prompt
    assert "Demonstrations (0):" in formatted_prompt


def test_format_comparison_shows_both_answers_for_same_question():
    formatted_comparison = format_comparison(
        "What is 10 + 5?",
        SimpleNamespace(answer="14"),
        SimpleNamespace(answer="15"),
    )

    assert "Question: What is 10 + 5?" in formatted_comparison
    assert "Before optimization: 14" in formatted_comparison
    assert "After optimization:  15" in formatted_comparison


def test_optimize_and_compare_compiles_all_rows_and_runs_same_question(
    monkeypatch,
):
    import dspy

    class FakeExample:
        def __init__(self, **values):
            self.values = values

        def with_inputs(self, *input_names):
            self.input_names = input_names
            return self

    class FakeProgram:
        def __init__(self, answer, instructions):
            self.answer = answer
            self.instructions = instructions
            self.questions = []
            self.demos = []

        def __call__(self, *, question):
            self.questions.append(question)
            return SimpleNamespace(answer=self.answer)

        def predictors(self):
            predictor = SimpleNamespace(
                signature=SimpleNamespace(instructions=self.instructions),
                demos=self.demos,
            )
            return [predictor]

    baseline_program = FakeProgram("before", "Baseline instructions")
    optimized_program = FakeProgram("after", "Optimized instructions")
    created_programs = []
    compiled_trainsets = []
    optimizer_options = {}

    def make_program(signature):
        assert signature == "question -> answer"
        created_programs.append(baseline_program)
        return baseline_program

    class FakeOptimizer:
        def __init__(self, **options):
            optimizer_options.update(options)

        def compile(self, program, *, trainset):
            assert program is baseline_program
            compiled_trainsets.append(trainset)
            return optimized_program

    monkeypatch.setattr(dspy, "Example", FakeExample)
    monkeypatch.setattr(dspy, "LM", Mock(return_value=object()))
    monkeypatch.setattr(dspy, "configure", Mock())
    monkeypatch.setattr(dspy, "ChainOfThought", make_program)
    monkeypatch.setattr(dspy, "BootstrapFewShot", FakeOptimizer)

    transformers_module = ModuleType("transformers")
    tokenizer = object()
    model = Mock()
    transformers_module.AutoTokenizer = SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer)
    )
    transformers_module.AutoModelForCausalLM = SimpleNamespace(
        from_pretrained=Mock(return_value=model)
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    records = [
        {"question": "What is 1 + 2?", "answer": "3"},
        {"question": "What is 4 + 5?", "answer": "9"},
        {"question": "What is 6 + 7?", "answer": "13"},
    ]
    question = "What is 20 + 22?"

    report = prompt_optimization.optimize_and_compare.local(records, question)

    assert len(created_programs) == 1
    assert len(compiled_trainsets) == 1
    assert [example.values for example in compiled_trainsets[0]] == records
    assert baseline_program.questions == [question]
    assert optimized_program.questions == [question]
    assert optimizer_options["max_bootstrapped_demos"] == 4
    assert optimizer_options["max_labeled_demos"] == 8
    assert optimizer_options["max_rounds"] == 5
    assert "Before optimization: before" in report
    assert "After optimization:  after" in report


def test_main_reads_jsonl_and_writes_modal_report(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "training.jsonl"
    output_path = tmp_path / "reports" / "prompt_report.txt"
    records = [
        {"question": "What is 1 + 2?", "answer": "3"},
        {"question": "What is 4 + 5?", "answer": "9"},
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    remote_call = Mock(return_value="prompt comparison report")
    monkeypatch.setattr(
        prompt_optimization,
        "optimize_and_compare",
        SimpleNamespace(remote=remote_call),
    )

    prompt_optimization.main(
        question="What is 20 + 22?",
        data_path=str(input_path),
        report_path=str(output_path),
    )

    remote_call.assert_called_once_with(records, "What is 20 + 22?")
    assert output_path.read_text(encoding="utf-8") == "prompt comparison report\n"
    assert "prompt comparison report" in capsys.readouterr().out
