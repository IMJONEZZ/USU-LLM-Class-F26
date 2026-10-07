"""Generation protocol and text scoring for the opt-in Llama evaluator."""

import json
from types import SimpleNamespace

import pytest
import torch

from src import evaluator as ev
from src import trainer as tr
from tests import test_evaluator as bert_fixtures
from tests import test_trainer as fixtures
from tests.test_trainer import tiny_base

examples = fixtures.examples
llama_tokenizer = fixtures.llama_tokenizer
metric_loader = bert_fixtures.metric_loader


class ScriptedGeneration(torch.nn.Module):
    def generate(self, **kwargs):
        self.arguments = kwargs
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        answers = [[4, 5, 1, 1], [1, 1, 1, 1], [6, 7, 8, 9]]
        return torch.cat(
            [kwargs["input_ids"], torch.tensor(answers[: len(kwargs["input_ids"])])],
            dim=1,
        )


def test_generation_prompt_only_left_padding_raw_answers(examples, llama_tokenizer):
    examples = examples[:3]
    examples[0]["prompt_ids"] = [2]
    model = ScriptedGeneration()
    rows = ev.generate_answers(
        examples, model, llama_tokenizer, batch_size=3, max_new_tokens=4
    )
    assert model.arguments["input_ids"].tolist() == [[1, 2], [2, 3], [2, 3]]
    assert model.arguments["attention_mask"].tolist() == [[0, 1], [1, 1], [1, 1]]
    assert model.arguments["max_new_tokens"] == 4
    assert [r["prediction"] for r in rows] == ["three four", "", "five six seven eight"]
    assert [r["status"] for r in rows] == ["ok", "empty", "length_limit"]
    assert rows[0]["exact"] is True
    assert rows[1]["exact"] is False
    assert rows[2]["generated_ids"] == [6, 7, 8, 9]
    assert rows[1]["format_issues"] == ["not_four_lexical_words"]
    assert rows[2]["format_issues"] == []


def test_real_tiny_causal_generation(examples, llama_tokenizer):
    model = tiny_base()
    rows = ev.generate_answers(
        examples, model, llama_tokenizer, batch_size=2, max_new_tokens=3
    )
    again = ev.generate_answers(
        examples, model, llama_tokenizer, batch_size=2, max_new_tokens=3
    )
    assert rows == again
    assert len(rows) == 5
    assert all(len(row["generated_ids"]) <= 3 for row in rows)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("  A\n B\tC ", "A B C"),
        ("A, b!", "A, b!"),
        ("", ""),
        ("one two three four explanation", "one two three four explanation"),
    ],
)
def test_normalization_only_whitespace(text, expected):
    assert ev.normalize_span(text) == expected


def test_empty_accounting_and_original_metric_settings(metric_loader):
    rows = [
        {
            "normalized_prediction": "three four",
            "normalized_reference": "three four",
            "exact": True,
        },
        {"normalized_prediction": "", "normalized_reference": "five", "exact": False},
    ]
    result = ev.generation_metrics(rows, metric_loader)
    assert result["example_count"] == 2
    assert result["empty_count"] == 1
    assert result["exact_span_match"] == 0.5
    assert result["rouge"]["rouge1"] == 0.25
    assert result["bertscore"]["f1"] == 0.15
    bleu = metric_loader("bleu").compute.call_args.kwargs
    assert bleu["predictions"] == ["three four", ""]
    assert bleu["max_order"] == 2 and bleu["smooth"]
    bs = metric_loader("bertscore").compute.call_args.kwargs
    assert bs["device"] == "cpu"
    assert bs["model_type"] == "roberta-base"
    assert not bs["idf"] and not bs["rescale_with_baseline"]
    assert ev.generation_metrics(rows[1:], metric_loader)["bleu"]["bleu"] == 0


def test_bad_generation_and_metrics(examples, llama_tokenizer):
    with pytest.raises(TypeError, match="string"):
        ev.normalize_span(None)
    with pytest.raises(ValueError, match="No predictions"):
        ev.generation_metrics([])
    with pytest.raises(ValueError, match="Need examples"):
        ev.generate_answers([], tiny_base(), llama_tokenizer)
    for output, message in [
        (torch.zeros((1,)), "Malformed"),
        (torch.tensor([[8, 8, 8]]), "preserve"),
    ]:
        model = SimpleNamespace(
            eval=lambda: None, generate=lambda output=output, **kw: output
        )
        with pytest.raises(ValueError, match=message):
            ev.generate_answers(examples[:1], model, llama_tokenizer)


def test_llama_cli_real_generation_fake_hub(
    tmp_path, monkeypatch, examples, llama_tokenizer
):
    import transformers

    prepared = tmp_path / "prepared"
    llama_tokenizer.save_pretrained(prepared / "tokenizer")
    data = {
        "model": "tiny",
        "revision": "immutable",
        "max_new_tokens": 3,
        "tokenizer_sha256": ev._digest(llama_tokenizer.backend_tokenizer.to_str()),
        "examples": {"test": examples},
    }
    data["signature"] = ev._digest(data)
    tr.write_json(prepared / "prepared.json", data)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **kw: tiny_base(),
    )
    monkeypatch.setattr(
        ev, "generation_metrics", lambda rows: {"example_count": len(rows)}
    )
    output = tmp_path / "baseline.json"
    args = [
        "llama",
        "--prepared",
        str(prepared),
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    result = ev.main(args)
    assert result["status"] == "complete"
    assert len(result["examples"]) == 5
    assert json.loads(output.read_text())["metadata"]["adapter"] is None
    with pytest.raises(SystemExit):
        ev.main(args)
    adapter = tmp_path / "adapter"
    tr.write_json(
        adapter / "run.json",
        {"signature": "wrong", "model": "tiny", "revision": "immutable"},
    )
    with pytest.raises(ValueError, match="do not match"):
        ev.main(args[:-1] + [str(tmp_path / "after.json"), "--adapter", str(adapter)])
    trained = tmp_path / "trained"
    training = tr.fit(
        fixtures.tiny_adapter(),
        examples,
        examples,
        trained,
        1,
        epochs=1,
        metadata={k: data[k] for k in ("signature", "model", "revision")},
    )
    final = ev.main(
        args[:-1] + [str(tmp_path / "final.json"), "--adapter", str(trained / "best")]
    )
    assert (
        final["metadata"]["training_run"]["checkpoint_epoch"] == training["best_epoch"]
    )
    assert [(r["prompt"], r["reference"]) for r in final["examples"]] == [
        (r["prompt"], r["reference"]) for r in result["examples"]
    ]
