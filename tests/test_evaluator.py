"""Exercise the notebook export using real data operations and fake models."""

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


def chat_generator(answers):
    answers = iter(answers)

    def respond(messages, **kwargs):
        return [
            {
                "generated_text": [
                    *messages,
                    {"role": "assistant", "content": next(answers)},
                ]
            }
        ]

    return Mock(side_effect=respond)


@pytest.fixture(params=[False, True], ids=["cpu", "cuda"])
def workflow(request, monkeypatch, capsys):
    """Run every script cell without downloads, real hardware, or local CSV files."""

    def module(name, **attributes):
        stub = ModuleType(name)
        stub.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, stub)
        return stub

    data = pd.DataFrame(
        {
            "Riddle": ["First riddle", "Second riddle"],
            "Answer": ["Echo", "Map"],
            "unused": [1, 2],
        },
        index=[9, 2],
    )
    read_csv = Mock(return_value=data)
    monkeypatch.setattr(pd, "read_csv", read_csv)
    vectors = {"Echo": [1, 0], "Map": [0, 1], "Globe": [3, 4]}
    embedder = Mock()
    embedder.encode.side_effect = lambda texts: np.array(
        [vectors[text] for text in texts]
    )
    constructor = Mock(return_value=embedder)
    module("sentence_transformers", SentenceTransformer=constructor)
    demo = chat_generator(["Piano"])
    generators = [
        chat_generator(answers)
        for answers in [["Echo", "Map"], ["Map", "Echo"], ["Echo", "Globe"]]
    ]
    pipeline = Mock(side_effect=[demo, *generators])
    utils = module("transformers.utils", logging=Mock())
    module("transformers", pipeline=pipeline, utils=utils)
    module("torch", cuda=Mock(is_available=Mock(return_value=request.param)))
    module("tqdm", tqdm=lambda rows, **kwargs: rows)
    pyplot = module("matplotlib.pyplot")
    module("matplotlib", pyplot=pyplot)
    path = Path(__file__).resolve().parents[1] / "src" / "evaluator.py"
    namespace = runpy.run_path(str(path), run_name="__main__")
    return (
        namespace,
        request.param,
        data,
        read_csv,
        constructor,
        pipeline,
        demo,
        generators,
    )


def test_workflow_loads_data_selects_devices_and_evaluates_models(workflow, capsys):
    namespace, cuda, data, read_csv, constructor, pipeline, demo, generators = workflow
    read_csv.assert_called_once()
    constructor.assert_called_once_with(
        "sentence-transformers/all-MiniLM-L6-v2", device="cuda" if cuda else "cpu"
    )
    expected_data = data[["Riddle", "Answer"]].rename(
        columns={"Riddle": "input", "Answer": "expected_answer"}
    )
    pd.testing.assert_frame_equal(namespace["riddles"], expected_data)
    assert pipeline.call_count == 4
    model_names = [
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]
    for call, name in zip(pipeline.call_args_list[1:], model_names, strict=True):
        assert call.args == ("text-generation",)
        assert call.kwargs == {"model": name, "device": 0 if cuda else -1}
    demo.assert_called_once()
    assert demo.call_args.kwargs == {"do_sample": True, "temperature": 1, "top_p": 0.9}
    for key, scores, answers, generator in zip(
        ["hf360m", "qwen_05b", "qwen_15b"],
        [[1.0, 1.0], [0.0, 0.0], [1.0, 0.8]],
        [["Echo", "Map"], ["Map", "Echo"], ["Echo", "Globe"]],
        generators,
        strict=True,
    ):
        mean, actual_scores, output = namespace[key]
        assert mean == pytest.approx(np.mean(scores))
        assert actual_scores == pytest.approx(scores)
        pd.testing.assert_frame_equal(
            output, expected_data.assign(generated_answer=answers, similarity=scores)
        )
        assert generator.call_count == 2
        for call, riddle in zip(generator.call_args_list, data["Riddle"], strict=True):
            messages = call.args[0]
            assert messages[0] == {
                "role": "system",
                "content": namespace["prompt_base"],
            }
            assert messages[-1] == {"role": "user", "content": riddle}
            assert not any(
                message["content"] in {"Echo", "Map"} for message in messages
            )
            assert call.kwargs == {"do_sample": False, "max_new_tokens": 16}
    printed = capsys.readouterr().out
    assert "Piano" in printed
    assert all(name in printed for name in model_names)


@pytest.mark.parametrize(
    ("vectors", "score"),
    [
        ([[3, 4], [6, 8]], 1.0),
        ([[2, 0], [0, 3]], 0.0),
        ([[3, 4], [-6, -8]], -1.0),
        ([[1, 0], [3, 4]], 0.6),
    ],
)
def test_evaluate_cosine_similarity_and_preserves_raw_answer(workflow, vectors, score):
    evaluate = workflow[0]["evaluate"]
    data = pd.DataFrame(
        {"input": ["Riddle"], "expected_answer": ["Reference"]}, index=[7]
    )
    original = data.copy(deep=True)
    generator = chat_generator(["Answer: a prediction"])
    embedder = Mock()
    embedder.encode.return_value = np.array(vectors)
    mean, scores, output = evaluate(generator, data, embedder, "Custom instruction")
    assert mean == pytest.approx(score)
    assert scores == pytest.approx([score])
    assert output["similarity"].tolist() == pytest.approx([score])
    assert output["generated_answer"].tolist() == ["Answer: a prediction"]
    assert output.index.tolist() == [7]
    embedder.encode.assert_called_once_with(["Reference", "Answer: a prediction"])
    pd.testing.assert_frame_equal(data, original)
