"""Offline checks of evaluation correctness, independent of pretrained quality.

The tokenizer has a tiny local WordPiece vocabulary. Hand-calculated answers
and a real, randomly initialized tiny BERT exercise the interfaces without Hub
downloads. Metric services are mocked; their configuration is asserted.
"""

import json
import runpy
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizer

from src import evaluator as ev


@pytest.fixture
def tokenizer():
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "don",
        "'",
        "’",
        "t",
        "c",
        "-",
        "3",
        "##po",
        ",",
        ".",
        "!",
        "?",
        "hello",
        "world",
        "cafe",
        "‐",
        "‑",
    ]
    return BertTokenizer(
        vocab={token: i for i, token in enumerate(tokens)}, do_lower_case=True
    )


def dialogue(text="one two three four five six seven eight", example_id="a"):
    return ev.Dialogue(example_id, "SPEAKER_NOT_IN_VOCAB", text)


def write_corpus(tmp_path, count=20, **extra):
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            [
                {"Character": f"speaker {i}", "Line": dialogue().text, **extra}
                for i in range(count)
            ]
        )
    )
    return path


class RecordingModel(torch.nn.Module):
    """Position-based logits with no dependence on batch padding or ordering."""

    def __init__(self, vocab_size, chosen_token=None):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0))
        self.vocab_size = vocab_size
        self.chosen_token = chosen_token
        self.calls = []
        self.config = SimpleNamespace(
            max_position_embeddings=512, _commit_hash="test-revision"
        )

    def forward(self, input_ids, attention_mask):
        assert not self.training
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled()
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, device=input_ids.device)
        for position in range(length):
            token = (
                self.chosen_token if self.chosen_token is not None else 5 + position % 8
            )
            logits[:, position, token] = 1
        return SimpleNamespace(logits=logits)


@pytest.fixture
def model(tokenizer):
    return RecordingModel(len(tokenizer))


@pytest.fixture
def metric_loader():
    def compute(name, **kwargs):
        count = len(kwargs["predictions"])
        if name == "rouge":
            return {"rouge1": [0.5] * count, "rougeL": [0.25] * count}
        if name == "bleu":
            return {"bleu": 0.125, "precisions": [0.5, 0.25]}
        return {
            "precision": [0.5] * count,
            "recall": [0.25] * count,
            "f1": [0.3] * count,
            "hashcode": "test-metric-hash",
        }

    metrics = {name: Mock() for name in ("rouge", "bleu", "bertscore")}
    for name, metric in metrics.items():
        metric.compute.side_effect = lambda name=name, **kwargs: compute(name, **kwargs)
    return Mock(side_effect=metrics.__getitem__)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        [],
        "text",
        4,
        [{}],
        [{"Character": "X"}],
        [{"Character": "X", "Line": 42}],
        [{"Character": None, "Line": "ok"}],
        ["line"],
    ],
)
def test_reject_malformed_corpus(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ev.load_dialogues(path)


def test_missing_or_invalid_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        ev.load_dialogues(tmp_path / "missing.json")
    path = tmp_path / "invalid.json"
    path.write_text("not json")
    with pytest.raises(ValueError):
        ev.load_dialogues(path)


def test_ids_survive_order_and_preserve_duplicates(tmp_path):
    records = [{"Character": "C", "Line": text} for text in ("hello", "world", "hello")]
    path = tmp_path / "data.json"
    path.write_text(json.dumps(records))
    first = ev.load_dialogues(path)
    path.write_text(json.dumps(records[::-1]))
    second = ev.load_dialogues(path)
    assert len({row.example_id for row in first}) == 3
    assert sorted(first, key=lambda r: r.example_id) == sorted(
        second, key=lambda r: r.example_id
    )


@pytest.mark.parametrize(
    "group_key", ["Scene", "scene", "Conversation", "conversation"]
)
def test_groups_are_qualified_by_episode(tmp_path, group_key):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"Character": "C", "Line": "hello", group_key: 1, "Episode": episode}
                for episode in (4, 5)
            ]
        )
    )
    rows = ev.load_dialogues(path)
    assert rows[0].group != rows[1].group


@pytest.mark.parametrize("missing", [None, "", [], True])
def test_partial_or_invalid_grouping_rejected(tmp_path, missing):
    path = tmp_path / "data.json"
    path.write_text(
        json.dumps(
            [
                {"Character": "C", "Line": "hello", "Scene": 1},
                {"Character": "C", "Line": "world", "Scene": missing},
            ]
        )
    )
    with pytest.raises(ValueError, match="Scene"):
        ev.load_dialogues(path)


def test_split_exact_membership_persistence_and_reordering(tmp_path):
    rows = ev.load_dialogues(write_corpus(tmp_path, 100))
    path = tmp_path / "nested" / "splits.json"
    first = ev.load_or_create_splits(rows, path)
    original_bytes = path.read_bytes()
    assert ev.Counter(first["assignments"].values()) == {
        "train": 80,
        "dev": 10,
        "test": 10,
    }
    assert set(first["assignments"]) == {row.example_id for row in rows}
    # Existing partition is authoritative even if a different masking seed is used.
    assert ev.load_or_create_splits(rows[::-1], path, seed=99) == first
    assert path.read_bytes() == original_bytes
    assert ev.load_or_create_splits(rows[::-1], tmp_path / "again.json") == first


def test_group_split_never_separates_a_group(tmp_path):
    rows = [replace(dialogue(example_id=str(i)), group=str(i // 3)) for i in range(30)]
    manifest = ev.load_or_create_splits(rows, tmp_path / "groups.json")
    assert manifest["split_unit"] == "group"
    for i in range(0, 30, 3):
        assert len({manifest["assignments"][str(j)] for j in range(i, i + 3)}) == 1


@pytest.mark.parametrize("count", [3, 4, 9, 10, 11])
def test_small_split_has_all_three_partitions(tmp_path, count):
    rows = [dialogue(example_id=str(i)) for i in range(count)]
    manifest = ev.load_or_create_splits(rows, tmp_path / "splits.json")
    assert set(manifest["assignments"].values()) == set(ev.SPLITS)


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [dialogue()],
        [dialogue(), dialogue("other")],
        [replace(dialogue(), example_id="")],
        [replace(dialogue(), text=None)],
        [dialogue(), replace(dialogue(example_id="b"), group="scene")],
    ],
)
def test_invalid_split_inputs(tmp_path, rows):
    with pytest.raises(ValueError):
        ev.load_or_create_splits(rows, tmp_path / "split.json")


def test_manifest_rejects_changed_corpus_without_overwriting(tmp_path):
    rows = ev.load_dialogues(write_corpus(tmp_path))
    path = tmp_path / "splits.json"
    ev.load_or_create_splits(rows, path)
    before = path.read_bytes()
    rows[0] = replace(rows[0], text="changed")
    with pytest.raises(ValueError, match="does not match"):
        ev.load_or_create_splits(rows, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "corruption", ["schema", "missing", "extra", "label", "empty_partition", "not_dict"]
)
def test_manifest_corruption(tmp_path, corruption):
    rows = ev.load_dialogues(write_corpus(tmp_path))
    path = tmp_path / "splits.json"
    manifest = ev.load_or_create_splits(rows, path)
    if corruption == "schema":
        manifest["schema_version"] = 99
    elif corruption == "missing":
        manifest["assignments"].pop(rows[0].example_id)
    elif corruption == "extra":
        manifest["assignments"]["extra"] = "test"
    elif corruption == "label":
        manifest["assignments"][rows[0].example_id] = "validation"
    elif corruption == "empty_partition":
        manifest["assignments"] = dict.fromkeys(manifest["assignments"], "train")
    else:
        manifest = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ev.load_or_create_splits(rows, path)


def test_reused_manifest_rejects_group_leakage(tmp_path):
    rows = [replace(dialogue(example_id=str(i)), group=str(i // 2)) for i in range(20)]
    path = tmp_path / "splits.json"
    manifest = ev.load_or_create_splits(rows, path)
    current = manifest["assignments"]["0"]
    manifest["assignments"]["1"] = "dev" if current != "dev" else "test"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="crosses"):
        ev.load_or_create_splits(rows, path)


@pytest.mark.parametrize("word", ["don't", "don’t", "C-3PO", "C‐3PO", "C‑3PO", "café"])
def test_lexical_words_keep_internal_apostrophes_hyphens_and_unicode(word):
    assert ev.WORD.findall(f"!{word},") == [word]


def test_mask_all_subwords_and_internal_punctuation_only(tokenizer):
    text = "one two don't C-3PO, three four five six."
    examples, report = ev.prepare_examples([dialogue(text)], tokenizer)
    example = examples[0]
    assert example.word_start == 2
    assert text[example.char_start : example.char_end] == "don't C-3PO, three four"
    original = tokenizer(text)["input_ids"]
    assert (
        list(example.reference_ids)
        == tokenizer("don't C-3PO, three four", add_special_tokens=False)["input_ids"]
    )
    assert len(example.reference_ids) > 4
    for i, token in enumerate(original):
        assert example.input_ids[i] == (
            tokenizer.mask_token_id if i in example.mask_positions else token
        )
    assert example.input_ids[0] == tokenizer.cls_token_id
    assert example.input_ids[-1] == tokenizer.sep_token_id
    assert example.input_ids[-2] == tokenizer.convert_tokens_to_ids(".")
    assert not (set(example.reference_ids) & set(tokenizer.all_special_ids))
    assert report["excluded_count"] == 0


def test_speaker_field_is_not_tokenized_or_masked(tokenizer):
    row = dialogue()
    examples, _ = ev.prepare_examples([row], tokenizer)
    assert len(examples[0].input_ids) == 10  # eight words + CLS/SEP only
    assert examples[0].text == row.text


def test_masking_independent_of_order_filtering_and_global_rng(tokenizer):
    rows = [
        dialogue(
            "one two three four five six seven eight nine ten eleven twelve", str(i)
        )
        for i in range(20)
    ]
    full, _ = ev.prepare_examples(rows, tokenizer)
    torch.manual_seed(123)
    reverse, _ = ev.prepare_examples(rows[::-1], tokenizer)
    subset, _ = ev.prepare_examples(rows[5:8], tokenizer)
    assert full == reverse
    mapping = {row.example_id: row for row in full}
    assert all(mapping[row.example_id] == row for row in subset)
    different, _ = ev.prepare_examples(rows, tokenizer, seed=1)
    assert any(
        a.mask_positions != b.mask_positions
        for a, b in zip(full, different, strict=True)
    )


def test_length_includes_special_tokens_and_exclusion_counts(tokenizer):
    rows = [
        dialogue("one " * 254, "exact"),
        dialogue("one " * 255, "long"),
        dialogue("one two three four five six seven", "short"),
        dialogue("", "empty"),
        dialogue("!!!", "punctuation"),
        dialogue("one two [MASK] three four five six seven", "special"),
        dialogue("unknown " * 8, "unknown"),
    ]
    examples, report = ev.prepare_examples(rows, tokenizer)
    assert [row.example_id for row in examples] == ["exact"]
    assert len(examples[0].input_ids) == 256
    assert report["excluded_by_reason"] == {
        "overlength": 1,
        "insufficient_words": 3,
        "no_maskable_span": 2,
    }
    assert report["source_count"] == 7
    assert report["eligible_count"] == 1
    assert report["excluded_count"] == 6
    assert len(report["excluded_examples"]) == 6


@pytest.mark.parametrize("setting", ["not_fast", "no_mask"])
def test_reject_incompatible_tokenizer(setting):
    tokenizer = SimpleNamespace(is_fast=setting != "not_fast", mask_token_id=None)
    with pytest.raises(ValueError, match="fast tokenizer"):
        ev.prepare_examples([dialogue()], tokenizer)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "8"])
def test_invalid_positive_parameters(tokenizer, model, value):
    with pytest.raises(ValueError):
        ev.prepare_examples([dialogue()], tokenizer, max_length=value)
    with pytest.raises(ValueError):
        ev.predict_spans([], model, tokenizer, batch_size=value)


@pytest.mark.parametrize("seed", [-1, True, 0.5, "0"])
def test_invalid_seeds(tokenizer, tmp_path, seed):
    with pytest.raises(ValueError):
        ev.prepare_examples([dialogue()], tokenizer, seed=seed)
    with pytest.raises(ValueError):
        ev.load_or_create_splits([dialogue()], tmp_path / "s.json", seed=seed)


def test_batching_preserves_order_references_and_final_partial_batch(tokenizer, model):
    rows = [dialogue("one " * length, str(length)) for length in (8, 9, 10, 11, 12)]
    examples, _ = ev.prepare_examples(rows, tokenizer)
    before = list(examples)
    predictions = ev.predict_spans(examples, model, tokenizer, batch_size=2)
    assert predictions == [
        [5 + position % 8 for position in row.mask_positions] for row in examples
    ]
    assert len(model.calls) == 3
    assert [ids.shape[0] for ids, _ in model.calls] == [2, 2, 1]
    for ids, attention in model.calls:
        assert torch.equal(ids == tokenizer.pad_token_id, attention == 0)
    assert ev.predict_spans(examples, model, tokenizer, batch_size=1) == predictions
    assert ev.predict_spans(examples, model, tokenizer, batch_size=100) == predictions
    assert examples == before


def test_special_argmax_is_not_filtered(tokenizer):
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    model = RecordingModel(len(tokenizer), tokenizer.sep_token_id)
    result = ev.evaluate_examples(
        examples, [dialogue()], model, tokenizer, include_text_metrics=False
    )
    assert result["bert"]["token_accuracy"] == 0
    assert result["examples"][0]["prediction_ids"] == [tokenizer.sep_token_id] * 4
    assert result["examples"][0]["prediction"] == "[SEP] [SEP] [SEP] [SEP]"


@pytest.mark.parametrize(
    "corruption", ["empty", "length", "negative", "outside", "duplicate", "unmasked"]
)
def test_invalid_masked_examples(tokenizer, model, corruption):
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    row = examples[0]
    modifications = {
        "empty": {"mask_positions": ()},
        "length": {"reference_ids": ()},
        "negative": {"mask_positions": (-1, 3, 4, 5)},
        "outside": {"mask_positions": (3, 4, 5, 100)},
        "duplicate": {"mask_positions": (3, 3, 4, 5)},
        "unmasked": {"input_ids": tuple(tokenizer(dialogue().text)["input_ids"])},
    }
    with pytest.raises(ValueError, match="Invalid masked|mask position"):
        ev.predict_spans([replace(row, **modifications[corruption])], model, tokenizer)


def test_empty_inference_and_left_padding_rejected(tokenizer, model):
    with pytest.raises(ValueError, match="No eligible"):
        ev.predict_spans([], model, tokenizer)
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    tokenizer.padding_side = "left"
    with pytest.raises(ValueError, match="right padding"):
        ev.predict_spans(examples, model, tokenizer)


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(1, 1),
        torch.zeros(2, 10, 34),
        torch.full((1, 10, 34), float("nan")),
        torch.full((1, 10, 34), float("inf")),
    ],
)
def test_invalid_model_logits(tokenizer, logits):
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    model = Mock(return_value=SimpleNamespace(logits=logits))
    with pytest.raises(ValueError, match="logits"):
        ev.predict_spans(examples, model, tokenizer)


def test_real_tiny_bert_offline(tokenizer):
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
            max_position_embeddings=64,
            pad_token_id=tokenizer.pad_token_id,
        )
    )
    rows = [dialogue(), dialogue("one two three four five six seven eight nine", "b")]
    examples, _ = ev.prepare_examples(rows, tokenizer)
    single = ev.predict_spans(examples, model, tokenizer, batch_size=1)
    batched = ev.predict_spans(examples, model, tokenizer, batch_size=2)
    assert single == batched
    assert all(
        len(prediction) == len(row.reference_ids)
        for prediction, row in zip(single, examples, strict=True)
    )


def test_micro_accuracy_is_not_mean_of_span_accuracies():
    result = ev.strict_metrics([[1], [2, 9, 9]], [[1], [2, 3, 4]])
    assert result == {
        "token_accuracy": 0.5,
        "exact_span_match": 0.5,
        "correct_tokens": 2,
        "masked_tokens": 4,
        "exact_spans": 1,
        "example_count": 2,
    }


@pytest.mark.parametrize(
    "predicted,reference,accuracy",
    [
        ([[1, 2]], [[1, 2]], 1.0),
        ([[0, 0]], [[1, 2]], 0.0),
        ([[1, 0]], [[1, 2]], 0.5),
    ],
)
def test_hand_calculated_accuracy(predicted, reference, accuracy):
    result = ev.strict_metrics(predicted, reference)
    assert result["token_accuracy"] == accuracy
    assert result["exact_span_match"] == (1.0 if accuracy == 1 else 0.0)


@pytest.mark.parametrize(
    "predicted,reference",
    [
        ([], []),
        ([[1]], []),
        ([], [[1]]),
        ([[]], [[]]),
        ([[1]], [[1, 2]]),
        ([[True]], [[1]]),
        ([[1]], [[False]]),
        ([[-1]], [[1]]),
        ([[1.0]], [[1]]),
    ],
)
def test_invalid_strict_inputs(predicted, reference):
    with pytest.raises(ValueError):
        ev.strict_metrics(predicted, reference)


def test_baseline_counts_training_tokens_only_and_breaks_ties(tokenizer):
    one, two = tokenizer.convert_tokens_to_ids(["one", "two"])
    assert ev.baseline_token([dialogue("two one")], tokenizer) == min(one, two)
    assert (
        ev.baseline_token([dialogue("two two one [MASK] [MASK] [MASK]")], tokenizer)
        == two
    )
    assert ev.baseline_token(
        [dialogue("!!!")], tokenizer
    ) == tokenizer.convert_tokens_to_ids("!")


@pytest.mark.parametrize("rows", [[], [dialogue("")], [dialogue("[MASK] [PAD]")]])
def test_baseline_empty_training_rejected(tokenizer, rows):
    with pytest.raises(ValueError):
        ev.baseline_token(rows, tokenizer)


def test_metric_wrappers_configuration_and_aggregation(metric_loader):
    predictions, references = ["one two", "three four"], ["one three", "three five"]
    result = ev.text_metrics(
        predictions, references, batch_size=3, metric_loader=metric_loader
    )
    assert result["rouge"] == {"rouge1": 0.5, "rougeL": 0.25}
    assert result["bleu"]["bleu"] == 0.125
    assert result["bertscore"] == {"precision": 0.5, "recall": 0.25, "f1": 0.3}
    assert result["bertscore_hash"] == "test-metric-hash"
    metric_loader("bleu").compute.assert_called_once_with(
        predictions=predictions,
        references=[[text] for text in references],
        max_order=2,
        smooth=True,
    )
    metric_loader("rouge").compute.assert_called_once_with(
        predictions=predictions,
        references=references,
        rouge_types=["rouge1", "rougeL"],
        use_stemmer=False,
        use_aggregator=False,
    )
    metric_loader("bertscore").compute.assert_called_once_with(
        predictions=predictions,
        references=references,
        model_type="roberta-base",
        lang="en",
        batch_size=3,
        device="cpu",
        idf=False,
        rescale_with_baseline=False,
    )


@pytest.mark.parametrize(
    "predictions,references",
    [([], []), (["a"], []), ([""], ["a"]), (["a"], [" "]), ([None], ["a"])],
)
def test_invalid_text_metric_inputs(predictions, references, metric_loader):
    with pytest.raises(ValueError):
        ev.text_metrics(predictions, references, metric_loader=metric_loader)
    metric_loader.assert_not_called()


def test_evaluation_freezes_predictions_and_only_scores_spans(
    tokenizer, model, metric_loader
):
    examples, _ = ev.prepare_examples([dialogue(), dialogue(example_id="b")], tokenizer)
    result = ev.evaluate_examples(
        examples,
        [dialogue("one one two")],
        model,
        tokenizer,
        batch_size=8,
        metric_loader=metric_loader,
    )
    assert len(model.calls) == 1
    assert result["baseline"]["token_id"] == tokenizer.convert_tokens_to_ids("one")
    rows = result["examples"]
    assert all(row["reference"] == "three four five six" for row in rows)
    assert all(
        row["masked_text"] == "one two [MISSING SPAN] seven eight" for row in rows
    )
    assert metric_loader("rouge").compute.call_args.kwargs["predictions"] == [
        row["prediction"] for row in rows
    ]
    assert (
        metric_loader("rouge").compute.call_args.kwargs["references"]
        == ["three four five six"] * 2
    )
    assert len(rows) == result["bert"]["example_count"] == 2
    # Everything needed for an audit is JSON serializable without custom encoders.
    assert json.loads(json.dumps(result))["bert"] == result["bert"]


@pytest.fixture
def cli(tmp_path, monkeypatch, tokenizer, model):
    import transformers

    path = write_corpus(tmp_path, 30)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForMaskedLM, "from_pretrained", Mock(return_value=model)
    )
    return [
        str(path),
        "--splits",
        str(tmp_path / "splits.json"),
        "--output",
        str(tmp_path / "results.json"),
    ]


def test_cli_full_test_report_and_roundtrip(cli, tmp_path, monkeypatch, metric_loader):
    import evaluate

    monkeypatch.setattr(evaluate, "load", metric_loader)
    result = ev.main(cli)
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["bert"] == result["bert"]
    assert saved["metadata"]["sample_count"] == 3
    assert saved["metadata"]["partition_counts"] == {"train": 24, "dev": 3, "test": 3}
    assert saved["metadata"]["resolved_revision"] == "test-revision"
    assert saved["metadata"]["metric_settings"]["bleu_max_order"] == 2
    assert saved["metadata"]["preparation"]["excluded_count"] == 0
    assert "transformers" in saved["metadata"]["versions"]


def test_cli_fixed_dev_subset_and_no_metric_download(cli, monkeypatch):
    monkeypatch.setattr(
        ev, "text_metrics", Mock(side_effect=AssertionError("must not load"))
    )
    first = ev.main(cli + ["--split", "dev", "--sample-size", "2", "--strict-only"])
    full = ev.main(cli + ["--split", "dev", "--strict-only"])
    assert first["metadata"]["sample_count"] == 2
    assert first["examples"] == full["examples"][:2]
    assert "text_metrics" not in first


@pytest.mark.parametrize(
    "args",
    [
        ["--sample-size", "1"],
        ["--batch-size", "0"],
        ["--seed", "-1"],
        ["--split", "dev", "--sample-size", "0"],
    ],
)
def test_cli_rejects_invalid_configuration(cli, args):
    with pytest.raises((ValueError, SystemExit)):
        ev.main(cli + args)


def test_cli_protects_source_and_manifest(cli):
    for path in (cli[0], cli[2]):
        with pytest.raises(SystemExit):
            ev.main(cli + ["--output", path])


def test_cli_no_eligible_examples(cli):
    with pytest.raises(ValueError, match="No eligible"):
        ev.main(cli + ["--max-length", "3"])


def test_cli_rejects_model_position_overflow(cli):
    with pytest.raises(ValueError, match="position embedding"):
        ev.main(cli + ["--max-length", "513"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", None),
        ("seed", True),
        ("seed", -1),
        ("split_unit", "group"),
        ("split_unit", None),
    ],
)
def test_manifest_metadata_validation(tmp_path, field, value):
    rows = ev.load_dialogues(write_corpus(tmp_path))
    path = tmp_path / "split.json"
    manifest = ev.load_or_create_splits(rows, path)
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ev.load_or_create_splits(rows, path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mask_positions", (True, 4, 5, 6)),
        ("mask_positions", (3.0, 4, 5, 6)),
        ("input_ids", (False, 5, 6, 4, 4, 4, 4, 11, 12, 3)),
        ("reference_ids", (-1, 8, 9, 10)),
    ],
)
def test_masked_example_rejects_invalid_numeric_types(tokenizer, model, field, value):
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    with pytest.raises(ValueError):
        ev.predict_spans([replace(examples[0], **{field: value})], model, tokenizer)


def test_no_pad_token_rejected(tokenizer, model):
    examples, _ = ev.prepare_examples([dialogue()], tokenizer)
    tokenizer.pad_token = None
    with pytest.raises(ValueError, match="pad token"):
        ev.predict_spans(examples, model, tokenizer)


def test_zero_scores_are_aggregated_correctly(metric_loader):
    metric_loader("rouge").compute.side_effect = None
    metric_loader("rouge").compute.return_value = {"rouge1": [0, 1], "rougeL": [0, 0.5]}
    result = ev.text_metrics(
        ["one", "two"], ["three", "two"], metric_loader=metric_loader
    )
    assert result["rouge"] == {"rouge1": 0.5, "rougeL": 0.25}


@pytest.mark.parametrize(
    "metric,label", [("rouge", "ROUGE"), ("bleu", "BLEU"), ("bertscore", "BERTScore")]
)
def test_missing_metric_result_is_reported_clearly(metric_loader, metric, label):
    backend = metric_loader(metric)
    backend.compute.side_effect = None
    backend.compute.return_value = None
    with pytest.raises(RuntimeError, match=f"{label} returned no result"):
        ev.text_metrics(["one two"], ["one three"], metric_loader=metric_loader)


def test_version_report_survives_missing_package_metadata(monkeypatch):
    def package_version(name):
        if name == "evaluate":
            raise ev.PackageNotFoundError(name)
        return "1.2.3"

    monkeypatch.setattr(ev, "version", package_version)
    versions = ev._versions()

    assert versions["python"] == sys.version.split()[0]
    assert "evaluate" not in versions
    # A missing entry must preserve earlier entries and allow later ones.
    assert versions["torch"] == "1.2.3"
    assert versions["numpy"] == "1.2.3"


def test_script_entry_point_writes_a_report(cli, tmp_path, monkeypatch, model, capsys):
    # Exercise __main__ using the same local tokenizer/model as the CLI tests.
    # run_path executes the file in-process so coverage sees the entry point.
    monkeypatch.setattr(
        sys,
        "argv",
        [ev.__file__, *cli, "--split", "dev", "--sample-size", "2", "--strict-only"],
    )
    runpy.run_path(ev.__file__, run_name="__main__")

    report = json.loads((tmp_path / "results.json").read_text())
    assert report["metadata"]["split"] == "dev"
    assert report["metadata"]["sample_count"] == 2
    assert report["bert"]["example_count"] == 2
    assert len(report["examples"]) == 2
    assert len(model.calls) == 1
    assert "text_metrics" not in report
    assert "Saved report to" in capsys.readouterr().out
