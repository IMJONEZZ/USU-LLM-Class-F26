import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import dspy
import numpy as np
import pytest
from datasets import Dataset

from src.prompt_optimization import (
    InvalidRiddleInstruction,
    SingleOutputAdapter,
    TransformersLM,
    describe_prompt,
    make_metric,
    prepare_examples,
    validate_riddle_instruction,
)


def test_plain_text_adapter_preserves_instructions_and_selected_demos():
    signature = dspy.Signature(
        "question -> answer", instructions="Answer in 1 to 3 words."
    )
    adapter = SingleOutputAdapter()
    demo = dspy.Example(question="What has hands?", answer="clock")
    messages = adapter.format(signature, [demo], {"question": "What can break?"})
    assert messages == [
        {"role": "system", "content": signature.instructions},
        {"role": "user", "content": demo.question},
        {"role": "assistant", "content": demo.answer},
        {"role": "user", "content": "What can break?"},
    ]
    # Regression: a valid plain answer needs neither field markers nor JSON.
    assert adapter.parse(signature, "  promise\n") == {"answer": "promise"}
    proposal_signature = dspy.Signature(
        "basic_instruction, task_demos -> proposed_instruction"
    )
    proposal = (
        "Solve each riddle by considering alternate meanings. Return only the answer."
    )
    assert adapter.parse(proposal_signature, proposal) == {
        "proposed_instruction": proposal
    }
    proposal_messages = adapter.format(
        proposal_signature,
        [],
        {"basic_instruction": "Solve riddles.", "task_demos": "An example riddle."},
    )
    assert "revised system prompt" in proposal_messages[0]["content"]
    assert (
        "Do not answer the example riddles yourself" in proposal_messages[-1]["content"]
    )


@pytest.mark.parametrize(
    "instruction",
    [
        "Propose an instruction that will be used to prompt a Language Model to provide possible answers to the following riddle:\n\nWhat is always before us yet unseen?",
        "Future",
        "Solve this riddle: What has keys but no locks?",
        "Write a prompt telling the model to solve riddles and return the answer.",
    ],
)
def test_rejects_meta_instructions_and_specific_riddles(instruction):
    signature = dspy.Signature("basic_instruction -> proposed_instruction")
    with pytest.raises(InvalidRiddleInstruction):
        SingleOutputAdapter().parse(signature, instruction)


def test_reusable_persona_instruction_is_allowed():
    instruction = (
        "You are a riddle-solving assistant who understands wordplay and puns."
    )
    assert validate_riddle_instruction(instruction) == instruction


def test_bad_instruction_proposal_is_retried_with_feedback():
    signature = dspy.Signature("basic_instruction -> proposed_instruction")
    valid = "Solve the user's riddle. Return only a short answer."
    with patch.object(
        dspy.Adapter,
        "__call__",
        side_effect=[
            InvalidRiddleInstruction("Do not propose another instruction."),
            [{"proposed_instruction": valid}],
        ],
    ) as call:
        result = SingleOutputAdapter()(
            None, {}, signature, [], {"basic_instruction": valid}
        )
    assert result == [{"proposed_instruction": valid}]
    assert call.call_count == 2
    assert "_proposal_feedback" in call.call_args.args[-1]
    assert validate_riddle_instruction(valid) == valid


def test_unusable_proposals_stop_after_three_attempts():
    signature = dspy.Signature("basic_instruction -> proposed_instruction")
    with (
        patch.object(
            dspy.Adapter, "__call__", side_effect=InvalidRiddleInstruction("bad")
        ) as call,
        pytest.raises(RuntimeError, match="after 3 attempts"),
    ):
        SingleOutputAdapter()(
            None, {}, signature, [], {"basic_instruction": "Solve riddles."}
        )
    assert call.call_count == 3


def test_split_is_reproducible_and_preserves_original_test_rows():
    dataset = Dataset.from_dict(
        {
            "input": [f"Riddle {i}" for i in range(100)],
            "expected_answer": [f"Answer {i}" for i in range(100)],
        }
    )
    examples, row_ids = prepare_examples(dataset)
    assert row_ids == prepare_examples(dataset)[1]
    assert [len(examples[name]) for name in ["train", "validation", "test"]] == [
        70,
        20,
        10,
    ]
    original_test = dataset.train_test_split(test_size=0.1, seed=42)["test"]
    assert [example.question for example in examples["test"]] == list(
        original_test["input"]
    )
    assert len(set(row_ids["train"] + row_ids["validation"] + row_ids["test"])) == 100
    assert set(examples["train"][0].inputs()) == {"question"}


def test_metric_matches_original_cosine_formula():
    class Embedder:
        def encode(self, texts, **kwargs):
            vectors = {
                "expected": np.array([3.0, 4.0]),
                "generated": np.array([4.0, 3.0]),
            }
            if isinstance(texts, list):
                return np.stack([self.encode(text) for text in texts])
            vector = vectors[texts]
            return vector / np.linalg.norm(vector)

    example = dspy.Example(question="riddle", answer="expected")
    metric = make_metric(Embedder(), [example])
    assert np.isclose(metric(example, dspy.Prediction(answer="generated")), 0.96)
    assert np.isclose(metric(example, dspy.Prediction(answer="expected")), 1.0)
    assert metric(example, dspy.Prediction(answer="")) == 0.0


def test_lm_copies_share_model_and_keep_generation_settings_separate():
    model, tokenizer = object(), object()
    task = TransformersLM(
        "local/qwen", model, tokenizer, temperature=0.0, max_tokens=64
    )
    proposer = task.copy(temperature=0.7, max_tokens=384)
    teacher = copy.deepcopy(task)
    assert proposer.hf_model is teacher.hf_model is model
    assert proposer.tokenizer is teacher.tokenizer is tokenizer
    assert proposer.inference_lock is task.inference_lock
    assert task.kwargs["temperature"] == 0.0
    assert proposer.kwargs["temperature"] == 0.7
    assert task.history is not proposer.history


def test_saved_prompt_can_be_inspected_and_formatted_after_reload(tmp_path):
    program = dspy.Predict(
        dspy.Signature("question -> answer", instructions="Solve riddles.")
    )
    program.demos = [dspy.Example(question="What has hands?", answer="clock")]
    path = tmp_path / "program.json"
    program.save(str(path))
    restored = dspy.Predict("question -> answer")
    restored.load(str(path))
    assert describe_prompt(restored) == describe_prompt(program)
    assert SingleOutputAdapter().format(
        restored.signature, restored.demos, {"question": "A new riddle"}
    ) == SingleOutputAdapter().format(
        program.signature, program.demos, {"question": "A new riddle"}
    )


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_transformers_backend_maps_generation_and_returns_only_new_tokens(temperature):
    model, tokenizer = Mock(), Mock()
    model.device = "cuda:0"
    model.config.max_position_embeddings = 128
    model.generate.return_value = np.array([[11, 12, 13, 21, 22]])
    batch = MagicMock()
    batch.to.return_value = batch
    batch.__getitem__.return_value = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), shape=(1, 3)
    )
    tokenizer.apply_chat_template.return_value = batch
    tokenizer.decode.return_value = "clock<stop>extra"
    tokenizer.pad_token_id = 0
    lm = TransformersLM("local/qwen", model, tokenizer)
    request = dspy.LMRequest(
        model="local/qwen",
        messages=[dspy.User("A riddle")],
        config=dspy.LMConfig(
            temperature=temperature, max_tokens=8, n=2, stop=["<stop>"]
        ),
    )

    response = lm.forward(request)

    batch.to.assert_called_once_with("cuda:0")
    assert model.generate.call_count == 2
    options = model.generate.call_args.kwargs
    assert options["do_sample"] == (temperature > 0)
    assert options["max_new_tokens"] == 8
    assert ("temperature" in options) == (temperature > 0)
    np.testing.assert_array_equal(tokenizer.decode.call_args.args[0], [21, 22])
    assert [output.parts[0].text for output in response.outputs] == ["clock", "clock"]
