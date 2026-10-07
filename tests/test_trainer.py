"""Real offline LoRA updates and task invariants; no model or corpus downloads."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from src import evaluator as ev
from src import trainer as tr


@pytest.fixture
def span_tokenizer():
    words = [
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
        ",",
    ]
    backend = Tokenizer(
        models.WordPiece({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


@pytest.fixture
def llama_tokenizer():
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[EOS]": 1,
                "one": 2,
                "two": 3,
                "three": 4,
                "four": 5,
                "five": 6,
                "six": 7,
                "seven": 8,
                "eight": 9,
                "nine": 10,
                "ten": 11,
                "eleven": 12,
                "twelve": 13,
                "THREE": 14,
                ",": 15,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        eos_token="[EOS]",
        pad_token="[EOS]",
    )


@pytest.fixture
def examples(llama_tokenizer):
    return [
        {
            "example_id": str(i),
            "prompt": "one two",
            "reference": ref,
            **tr.encode_answer("one two", ref, llama_tokenizer),
        }
        for i, ref in enumerate(
            ("three four", "five", "six seven eight", "nine ten", "eleven")
        )
    ]


def tiny_base():
    torch.manual_seed(123)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=512,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
            attention_dropout=0.0,
        )
    )


def tiny_adapter():
    return get_peft_model(
        tiny_base(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )


def test_duplicate_priority_preserves_manifest_and_exact_matching():
    rows = [
        ev.Dialogue(str(i), speaker, text)
        for i, (speaker, text) in enumerate(
            [
                ("A", "same"),
                ("B", "same"),
                ("C", "same"),
                ("D", "dev/train"),
                ("E", "dev/train"),
                ("F", "Same"),
                ("G", "same "),
                ("H", "same"),
            ]
        )
    ]
    assignments = dict(
        zip(
            [r.example_id for r in rows],
            ["train", "dev", "test", "train", "dev", "train", "dev", "test"],
        )
    )
    manifest = {"assignments": assignments}
    original = deepcopy(manifest)
    kept, audit = tr.isolated_partitions(rows, manifest)
    assert manifest == original
    assert [r.example_id for r in kept["test"]] == ["2", "7"]
    assert [r.example_id for r in kept["dev"]] == ["4", "6"]
    assert [r.example_id for r in kept["train"]] == ["5"]
    assert audit["original_records"] == {"train": 3, "dev": 3, "test": 2}
    assert audit["retained_records"] == {"train": 1, "dev": 2, "test": 2}
    assert {r["example_id"] for r in audit["exclusions"]} == {"0", "1", "3"}
    assert audit["exclusions"][-2]["conflicting_splits"] == ["test", "dev"]


def test_preparation_reuses_spans_and_raw_case_punctuation(
    span_tokenizer, llama_tokenizer
):
    rows = [
        ev.Dialogue("a", "LABEL", "one two THREE, four five six seven eight"),
        ev.Dialogue(
            "b",
            "LABEL",
            "one two three four five six seven eight nine ten eleven twelve",
        ),
    ]
    prepared, audit = tr.prepare_partition(rows, span_tokenizer, llama_tokenizer)
    legacy, _ = ev.prepare_examples(rows, span_tokenizer)
    assert (
        prepared
        == tr.prepare_partition(list(reversed(rows)), span_tokenizer, llama_tokenizer)[
            0
        ]
    )
    for row, old in zip(prepared, legacy, strict=True):
        assert row["reference"] == old.text[old.char_start : old.char_end]
        assert row["char_start"] == old.char_start
        assert "LABEL" not in row["prompt"]
    assert prepared[0]["reference"] == "THREE, four five six"
    assert "one two [MISSING SPAN] seven eight" in prepared[0]["prompt"]
    assert audit["eligible_examples"] == 2


def test_preparation_exclusions(span_tokenizer, llama_tokenizer):
    rows = [
        ev.Dialogue("short", "X", "one two"),
        ev.Dialogue("long", "X", "one two three four five six seven eight"),
    ]
    result, audit = tr.prepare_partition(
        rows, span_tokenizer, llama_tokenizer, max_length=5
    )
    assert result == []
    assert audit["retained_records"] == 2
    assert audit["bert_eligible_examples"] == 1
    assert audit["excluded_by_reason"] == {
        "insufficient_words": 1,
        "llama_overlength": 1,
    }
    assert tr.prepare_partition([], span_tokenizer, llama_tokenizer)[0] == []


def test_compositional_boundary_and_eos_padding(examples, llama_tokenizer):
    batch = tr.collate_answers(examples, llama_tokenizer.pad_token_id)
    for i, row in enumerate(examples):
        boundary, length = len(row["prompt_ids"]), len(row["input_ids"])
        assert batch["labels"][i, :boundary].tolist() == [-100] * boundary
        assert (
            batch["labels"][i, boundary:length].tolist() == row["input_ids"][boundary:]
        )
        assert batch["labels"][i, length - 1] == llama_tokenizer.eos_token_id
        assert (batch["labels"][i, length:] == -100).all()
        assert batch["attention_mask"][i].sum() == length
        assert (batch["input_ids"][i, length:] == llama_tokenizer.eos_token_id).all()
    with pytest.raises(ValueError, match="EOS"):
        tr.encode_answer("one", "two", SimpleNamespace(eos_token_id=None))
    with pytest.raises(ValueError, match="nonempty"):
        tr.encode_answer("", "two", llama_tokenizer)
    with pytest.raises(ValueError, match="empty"):
        tr.collate_answers([], 1)
    with pytest.raises(ValueError, match="aligned"):
        tr.collate_answers([{"input_ids": [1, 2], "labels": [-100]}], 1)


def test_loader_reproducibility_and_partial_batch(examples):
    a = list(tr.answer_loader(examples, 1, 2, True, 42))
    b = list(tr.answer_loader(examples, 1, 2, True, 42))
    assert [r["input_ids"].shape[0] for r in a] == [2, 2, 1]
    assert all(
        torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, b, strict=True)
    )


def test_real_training_frozen_base_checkpoint_reload(examples, tmp_path):
    model = tiny_adapter()
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    inputs = tr.collate_answers(examples[:2], 1)
    saved_logits = []
    save = model.save_pretrained

    def save_with_logits(path):
        with torch.no_grad():
            saved_logits.append(model(**inputs).logits.clone())
        save(path)

    model.save_pretrained = save_with_logits
    result = tr.fit(
        model,
        examples,
        examples[:2],
        tmp_path,
        1,
        epochs=2,
        batch_size=2,
        accumulation=2,
        learning_rate=0.02,
        metadata={"model": "tiny-local"},
    )
    assert result["best_epoch"] in (1, 2)
    assert result["best_dev_loss"] == min(r["dev_loss"] for r in result["history"])
    assert [r["optimizer_steps"] for r in result["history"]] == [2, 2]
    assert all(torch.isfinite(torch.tensor(r["train_loss"])) for r in result["history"])
    assert result["peak_allocated_bytes"] is None
    assert result["training_seconds"] > 0
    changed = []
    for name, p in model.named_parameters():
        if "lora_" not in name:
            assert torch.equal(initial[name], p)
        else:
            changed.append(not torch.equal(initial[name], p))
    assert any(changed)
    reloaded = PeftModel.from_pretrained(tiny_base(), tmp_path / "best")
    reloaded_again = PeftModel.from_pretrained(tiny_base(), tmp_path / "best")
    inputs = tr.collate_answers(examples[:2], 1)
    reloaded.eval()
    reloaded_again.eval()
    with torch.no_grad():
        assert torch.equal(reloaded(**inputs).logits, reloaded_again(**inputs).logits)
        assert torch.equal(reloaded(**inputs).logits, saved_logits[-1])
    saved = json.loads((tmp_path / "best/run.json").read_text())
    assert saved["model"] == "tiny-local"
    assert saved["checkpoint_epoch"] == result["best_epoch"]


def test_accumulation_matches_one_batch_token_weighted(examples):
    first = tiny_adapter()
    second = tiny_adapter()
    p1, _ = tr.adapter_parameters(first)
    p2, _ = tr.adapter_parameters(second)
    a = tr.train_epoch(
        first, tr.answer_loader(examples, 1, 5), torch.optim.SGD(p1, lr=0.01), p1
    )
    b = tr.train_epoch(
        second,
        tr.answer_loader(examples, 1, 1),
        torch.optim.SGD(p2, lr=0.01),
        p2,
        accumulation=5,
    )
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-6)
    for x, y in zip(p1, p2, strict=True):
        assert torch.allclose(x, y, atol=1e-6)


def test_overflow_retries_entire_group_without_changing_update(examples):
    reference, retried = tiny_adapter(), tiny_adapter()
    p1, _ = tr.adapter_parameters(reference)
    p2, _ = tr.adapter_parameters(retried)
    before = {n: p.detach().clone() for n, p in retried.named_parameters()}
    backward_calls = 0

    def overflow_second_microbatch(gradient):
        nonlocal backward_calls
        backward_calls += 1
        return gradient * float("inf") if backward_calls == 2 else gradient

    hook = p2[-1].register_hook(overflow_second_microbatch)
    expected = tr.train_epoch(
        reference,
        tr.answer_loader(examples, 1),
        torch.optim.AdamW(p1),
        p1,
        accumulation=2,
        scaler=torch.amp.GradScaler("cpu", init_scale=8),
    )
    scaler = torch.amp.GradScaler("cpu", init_scale=8)
    optimizer = torch.optim.AdamW(p2)
    actual = tr.train_epoch(
        retried,
        tr.answer_loader(examples, 1),
        optimizer,
        p2,
        accumulation=2,
        scaler=scaler,
    )
    hook.remove()
    assert actual["overflow_retries"] == 1
    assert actual["loss_scale"] == 4
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 3
    assert actual["answer_tokens"] == expected["answer_tokens"]
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    for first, second in zip(p1, p2, strict=True):
        assert torch.allclose(first, second, atol=1e-6)
    for name, parameter in retried.named_parameters():
        if "lora_" not in name:
            assert torch.equal(parameter, before[name])
    # Reusing the scaler in another epoch retains its calibrated scale.
    second_epoch = tr.train_epoch(
        retried,
        tr.answer_loader(examples, 1),
        optimizer,
        p2,
        scaler=scaler,
    )
    assert second_epoch["loss_scale"] == 4


@pytest.mark.parametrize("scaling", [False, True])
def test_persistent_nonfinite_gradients_never_update_weights(examples, scaling):
    model = tiny_adapter()
    parameters, _ = tr.adapter_parameters(model)
    before = [p.detach().clone() for p in parameters]
    hook = parameters[-1].register_hook(lambda gradient: gradient * float("inf"))
    optimizer = torch.optim.AdamW(parameters)
    with pytest.raises(RuntimeError, match="Non-finite gradients"):
        tr.train_epoch(
            model,
            tr.answer_loader(examples[:1], 1),
            optimizer,
            parameters,
            scaler=torch.amp.GradScaler("cpu", enabled=scaling),
        )
    hook.remove()
    assert not optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(before, parameters, strict=True))


def test_fit_keeps_one_scaler_across_epochs(examples, tmp_path, monkeypatch):
    original = tr.train_epoch
    scalers = []

    def capture_scaler(*args, **kwargs):
        scalers.append(kwargs["scaler"])
        return original(*args, **kwargs)

    monkeypatch.setattr(tr, "train_epoch", capture_scaler)
    tr.fit(tiny_adapter(), examples, examples, tmp_path, 1, epochs=2)
    assert len(scalers) == 2
    assert scalers[0] is scalers[1]


def test_dev_selects_earlier_checkpoint_not_last(examples, tmp_path, monkeypatch):
    model = tiny_adapter()
    losses = iter([5.0, 2.0, 3.0])
    monkeypatch.setattr(tr, "development_loss", lambda *a: next(losses))
    result = tr.fit(model, examples, examples, tmp_path, 1, epochs=2)
    assert result["best_epoch"] == 1
    assert result["best_dev_loss"] == 2.0
    assert (
        json.loads((tmp_path / "training.json").read_text())["history"][-1]["dev_loss"]
        == 3.0
    )
    best = PeftModel.from_pretrained(tiny_base(), tmp_path / "best")
    assert any(
        not torch.equal(p, dict(best.named_parameters())[n])
        for n, p in model.named_parameters()
        if "lora_" in n
    )


def test_loss_is_model_shift_and_token_weighted(examples):
    model = tiny_adapter().eval()
    batch = tr.collate_answers(examples, 1)
    with torch.no_grad():
        output = model(**batch)
        manual = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].reshape(-1, 16), batch["labels"][:, 1:].reshape(-1)
        )
    assert float(output.loss) == pytest.approx(float(manual))
    assert tr.development_loss(model, tr.answer_loader(examples, 1)) == pytest.approx(
        float(manual), abs=1e-6
    )


def test_training_rejects_bad_inputs(examples, tmp_path):
    with pytest.raises(ValueError, match="Only LoRA"):
        tr.adapter_parameters(tiny_base())
    with pytest.raises(ValueError, match="epochs"):
        tr.fit(tiny_adapter(), examples, examples, tmp_path, 1, epochs=0)
    with pytest.raises(ValueError, match="learning_rate"):
        tr.fit(
            tiny_adapter(), examples, examples, tmp_path, 1, learning_rate=float("nan")
        )
    with pytest.raises(ValueError, match="development"):
        tr.development_loss(tiny_adapter(), [])
    model = tiny_adapter()
    params, _ = tr.adapter_parameters(model)
    opt = torch.optim.AdamW(params)
    with pytest.raises(ValueError, match="training"):
        tr.train_epoch(model, [], opt, params)
    batch = tr.collate_answers(examples, 1)
    model.base_model.model.model.embed_tokens.weight.data.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        tr._loss(model, batch, "cpu")


def test_short_feasibility_loop_really_updates(examples):
    model = tiny_adapter()
    params, _ = tr.adapter_parameters(model)
    before = [p.detach().clone() for p in params]
    stats = tr.train_epoch(
        model,
        tr.answer_loader(examples, 1),
        torch.optim.AdamW(params),
        params,
        max_steps=2,
    )
    assert stats["optimizer_steps"] == 2
    assert any(not torch.equal(x, y) for x, y in zip(before, params, strict=True))


def test_prepared_integrity(tmp_path):
    data = {"hello": "world"}
    data["signature"] = ev._digest(data)
    tr.write_json(tmp_path / "prepared.json", data)
    assert tr.load_prepared(tmp_path) == data
    data["hello"] = "changed"
    tr.write_json(tmp_path / "prepared.json", data)
    with pytest.raises(ValueError, match="changed"):
        tr.load_prepared(tmp_path)


def test_cli_prepare_persisted_inputs(
    tmp_path, monkeypatch, span_tokenizer, llama_tokenizer
):
    import transformers

    rows = [
        {
            "Character": str(i),
            "Line": f"one two three four five six seven eight {suffix}",
        }
        for i, suffix in enumerate(["nine", "ten", "eleven"])
    ]
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps(rows))
    splits = tmp_path / "splits.json"
    ev.load_or_create_splits(ev.load_dialogues(corpus), splits)
    original = splits.read_bytes()
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(_commit_hash="pinned"),
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda name, **kw: span_tokenizer if name == ev.MODEL_ID else llama_tokenizer,
    )
    out = tmp_path / "prepared"
    assert (
        tr.main(
            [
                "prepare",
                "--corpus",
                str(corpus),
                "--splits",
                str(splits),
                "--output",
                str(out),
            ]
        )
        == 0
    )
    data = tr.load_prepared(out)
    assert all(len(r) == 1 for r in data["examples"].values())
    assert data["revision"] == "pinned"
    assert splits.read_bytes() == original
    with pytest.raises(SystemExit):
        tr.main(["prepare", "--output", str(out)])


def test_cli_missing_manifest_no_regeneration(tmp_path):
    out = tmp_path / "failed"
    missing = tmp_path / "missing.json"
    assert tr.main(["prepare", "--splits", str(missing), "--output", str(out)]) == 1
    assert not missing.exists()
    assert (
        json.loads((out / "failure.json").read_text())["error_type"]
        == "FileNotFoundError"
    )


def test_cpu_gpu_command_failure_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "gpu"
    assert tr.main(["feasibility", "--output", str(out)]) == 1
    failure = json.loads((out / "failure.json").read_text())
    assert "CUDA unavailable" in failure["error"]
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    assert (
        tr.safe_error(ValueError("secret-value hf_abcdefgh")) == "[REDACTED] [REDACTED]"
    )
    assert "peft" in tr.versions()


@pytest.mark.parametrize(
    "args",
    [
        ["train", "--epochs", "0"],
        ["train", "--learning-rate", "nan"],
        ["prepare", "--seed", "-1"],
    ],
)
def test_bad_cli_options(args):
    with pytest.raises((ValueError, SystemExit)):
        tr.main(args)
