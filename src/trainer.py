"""Answer-only LoRA training. GPU dependencies are imported only by GPU commands."""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter
from contextlib import nullcontext
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from src import evaluator as ev
from src.dataloader import seeded_loader

MODEL = "meta-llama/Llama-3.2-1B"
PROMPT = (
    "Recover the four missing words. Return only the missing text.\n\n"
    "Dialogue:\n{prefix}[MISSING SPAN]{suffix}\n\nMissing text:\n"
)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def versions():
    result = ev._versions()
    for name in (
        "peft",
        "unsloth",
        "unsloth_zoo",
        "trl",
        "accelerate",
        "xformers",
        "triton",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            pass
    return result


def isolated_partitions(dialogues, manifest):
    """Exact raw Line equality: case, punctuation and whitespace are significant.

    Use ORIGINAL memberships for comparisons; do not move or rewrite any record.
    Within-partition duplicates remain, and all test records remain unchanged.
    """
    original = {
        s: [r for r in dialogues if manifest["assignments"][r.example_id] == s]
        for s in ev.SPLITS
    }
    texts = {s: {r.text for r in rows} for s, rows in original.items()}
    kept, excluded = {}, []
    for split, higher in (("test", ()), ("dev", ("test",)), ("train", ("test", "dev"))):
        kept[split] = []
        for row in original[split]:
            conflicts = [s for s in higher if row.text in texts[s]]
            if conflicts:
                excluded.append(
                    {
                        "example_id": row.example_id,
                        "split": split,
                        "reason": "exact_dialogue_in_higher_priority_split",
                        "conflicting_splits": conflicts,
                    }
                )
            else:
                kept[split].append(row)
    return kept, {
        "matching_rule": "Exact raw Line string equality; no normalization; ignore speaker",
        "priority": ["test", "dev", "train"],
        "original_records": {s: len(rows) for s, rows in original.items()},
        "retained_records": {s: len(rows) for s, rows in kept.items()},
        "excluded_records": dict(Counter(r["split"] for r in excluded)),
        "exclusions": excluded,
    }


def encode_answer(prompt, reference, tokenizer):
    """Explicit compositional boundary, shared with generation (no double shift).

    Encode the entire prompt once with BOS, then the answer without special
    tokens, then EOS. Separate encoding prevents a BPE merge across the boundary.
    Padding is masked by position, never by token ID.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("A tokenizer with an EOS token is required")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(reference, add_special_tokens=False)
    if not prompt_ids or not answer_ids:
        raise ValueError("Prompt and answer must both tokenize to nonempty sequences")
    ids = prompt_ids + answer_ids + [tokenizer.eos_token_id]
    return {
        "prompt_ids": prompt_ids,
        "input_ids": ids,
        "labels": [-100] * len(prompt_ids) + answer_ids + [tokenizer.eos_token_id],
    }


def prepare_partition(
    rows, span_tokenizer, tokenizer, seed=0, max_length=256, max_new_tokens=32
):
    # Keep Assignment 3's BERT eligibility and candidate ordering EXACTLY, but
    # extract references from raw characters, never from uncased BERT token IDs.
    if rows:
        masked, audit = ev.prepare_examples(rows, span_tokenizer, seed, 256)
    else:
        masked, audit = (
            [],
            {
                "source_count": 0,
                "eligible_count": 0,
                "excluded_count": 0,
                "excluded_by_reason": {},
                "excluded_examples": [],
            },
        )
    examples, excluded = [], list(audit["excluded_examples"])
    for row in masked:
        prompt = PROMPT.format(
            prefix=row.text[: row.char_start], suffix=row.text[row.char_end :]
        )
        reference = row.text[row.char_start : row.char_end]
        encoded = encode_answer(prompt, reference, tokenizer)
        # Reserve a fixed generation budget, independent of reference length.
        if (
            len(encoded["input_ids"]) > max_length
            or len(encoded["prompt_ids"]) + max_new_tokens > max_length
        ):
            excluded.append(
                {"example_id": row.example_id, "reason": "llama_overlength"}
            )
            continue
        examples.append(
            {
                "example_id": row.example_id,
                "prompt": prompt,
                "reference": reference,
                "char_start": row.char_start,
                "char_end": row.char_end,
                **encoded,
            }
        )
    return examples, {
        "retained_records": len(rows),
        "bert_eligible_examples": len(masked),
        "eligible_examples": len(examples),
        "excluded_examples": excluded,
        "excluded_by_reason": dict(Counter(r["reason"] for r in excluded)),
    }


def collate_answers(rows, pad_token_id):
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(r["input_ids"]) for r in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        length = len(row["input_ids"])
        if length != len(row["labels"]) or not any(
            x != -100 for x in row["labels"][1:]
        ):
            raise ValueError("Each sequence needs aligned labels and a causal target")
        ids.append(row["input_ids"] + [pad_token_id] * (width - length))
        masks.append([1] * length + [0] * (width - length))
        labels.append(row["labels"] + [-100] * (width - length))
    return {
        name: torch.tensor(values, dtype=torch.long)
        for name, values in (
            ("input_ids", ids),
            ("attention_mask", masks),
            ("labels", labels),
        )
    }


def answer_loader(rows, pad_token_id, batch_size=1, shuffle=False, seed=0):
    return seeded_loader(
        rows,
        batch_size,
        shuffle,
        seed,
        partial(collate_answers, pad_token_id=pad_token_id),
    )


def adapter_parameters(model):
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n, _ in trainable):
        raise ValueError("Only LoRA adapter parameters may require gradients")
    # FP32 adapters permit GradScaler unscaling; the frozen GPU base stays FP16.
    for _, parameter in trainable:
        parameter.data = parameter.data.float()
    return [p for _, p in trainable], {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for _, p in trainable),
        "trainable_names": [n for n, _ in trainable],
    }


def _autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.float16)
        if str(device).startswith("cuda")
        else nullcontext()
    )


def _loss(model, batch, device):
    with _autocast(device):
        loss = model(
            **{k: v.to(device) for k, v in batch.items()}, use_cache=False
        ).loss
    if not torch.isfinite(loss):
        raise ValueError("Non-finite loss")
    return loss


def development_loss(model, loader, device="cpu"):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            tokens = int((batch["labels"][:, 1:] != -100).sum())
            total += float(_loss(model, batch, device)) * tokens
            count += tokens
    if not count:
        raise ValueError("No development targets")
    return total / count


def train_epoch(
    model,
    loader,
    optimizer,
    parameters,
    accumulation=1,
    device="cpu",
    max_steps=None,
    *,
    scaler=None,
):
    """Token-weighted accumulated gradients, including the final partial group."""
    ev._integer("accumulation", accumulation, 1)
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler("cuda", enabled=str(device).startswith("cuda"))
    total, count, steps, overflows = 0.0, 0, 0, 0
    iterator = iter(loader)
    while True:
        group = []
        for _ in range(accumulation):
            batch = next(iterator, None)
            if batch is None:
                break
            group.append(batch)
        if not group:
            break
        # Four lexical words can have different token counts. Weight each batch
        # by its supervised tokens to match one combined batch's mean loss.
        counts = [int((b["labels"][:, 1:] != -100).sum()) for b in group]
        retries = 0
        while True:
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for batch, tokens in zip(group, counts, strict=True):
                loss = _loss(model, batch, device)
                scaler.scale(loss * tokens / sum(counts)).backward()
                group_loss += float(loss.detach()) * tokens
            scaler.unscale_(optimizer)
            finite = bool(
                torch.stack(
                    [
                        torch.isfinite(p.grad).all()
                        for p in parameters
                        if p.grad is not None
                    ]
                ).all()
            )
            if finite:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                break
            if not scaler.is_enabled():
                raise RuntimeError("Non-finite gradients without loss scaling")
            # unscale_ recorded the overflow: step skips the optimizer update,
            # and update reduces the scale. Replay the whole accumulation group
            # so neither examples nor successful feasibility steps are lost.
            scaler.step(optimizer)
            scaler.update()
            overflows += 1
            retries += 1
            print(
                json.dumps(
                    {
                        "status": "gradient_overflow_retry",
                        "retry": retries,
                        "loss_scale": scaler.get_scale(),
                    }
                ),
                flush=True,
            )
            if retries >= 32:
                raise RuntimeError(
                    "Non-finite gradients persisted after 32 loss-scale reductions"
                )
        total += group_loss
        count += sum(counts)
        steps += 1
        if steps == 1 or steps % 25 == 0:
            print(
                json.dumps({"optimizer_steps": steps, "train_loss": total / count}),
                flush=True,
            )
        if max_steps is not None and steps >= max_steps:
            break
    if not count:
        raise ValueError("No training targets")
    return {
        "loss": total / count,
        "optimizer_steps": steps,
        "answer_tokens": count,
        "overflow_retries": overflows,
        "loss_scale": scaler.get_scale(),
    }


def fit(
    model,
    training,
    development,
    output,
    pad_token_id,
    *,
    epochs=3,
    batch_size=1,
    accumulation=8,
    learning_rate=2e-4,
    seed=0,
    device="cpu",
    max_steps=None,
    metadata=None,
):
    """Shared CPU/GPU loop. Select only post-epoch development loss, never test."""
    for name, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("accumulation", accumulation),
    ):
        ev._integer(name, value, 1)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    parameters, counts = adapter_parameters(model)
    print(json.dumps(counts), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    train = answer_loader(training, pad_token_id, batch_size, True, seed)
    dev = answer_loader(development, pad_token_id, batch_size)
    result = {
        "parameters": counts,
        "metadata": metadata or {},
        "history": [],
        "initial_dev_loss": development_loss(model, dev, device),
        "best_epoch": None,
        "best_dev_loss": None,
        "training_seconds": 0.0,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "timing_scope": "Training epoch loops including batching, backward and optimizer; excludes loading, dev evaluation and checkpoint saves",
    }
    gpu = str(device).startswith("cuda")
    # Keep the calibrated scale across epochs instead of repeating overflows.
    scaler = torch.amp.GradScaler("cuda", enabled=gpu)
    for epoch in range(1, epochs + 1):
        print(
            json.dumps({"status": "training_epoch", "epoch": epoch, "epochs": epochs}),
            flush=True,
        )
        if gpu:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        stats = train_epoch(
            model,
            train,
            optimizer,
            parameters,
            accumulation,
            device,
            max_steps,
            scaler=scaler,
        )
        if gpu:
            torch.cuda.synchronize()
        result["training_seconds"] += time.perf_counter() - start
        if gpu:
            for key, measured in (
                ("peak_allocated_bytes", torch.cuda.max_memory_allocated()),
                ("peak_reserved_bytes", torch.cuda.max_memory_reserved()),
            ):
                result[key] = max(result[key] or 0, measured)
        dev_loss = development_loss(model, dev, device)
        result["history"].append(
            {
                "epoch": epoch,
                "train_loss": stats["loss"],
                "dev_loss": dev_loss,
                "optimizer_steps": stats["optimizer_steps"],
                "answer_tokens": stats["answer_tokens"],
                "overflow_retries": stats["overflow_retries"],
                "loss_scale": stats["loss_scale"],
            }
        )
        # Strict improvement keeps the earlier checkpoint on a tie; the last
        # epoch is not automatically best, and test scores never select it.
        if result["best_dev_loss"] is None or dev_loss < result["best_dev_loss"]:
            result.update(best_epoch=epoch, best_dev_loss=dev_loss)
            model.save_pretrained(output / "best")
            write_json(
                output / "best" / "run.json",
                {
                    **result["metadata"],
                    "checkpoint_epoch": epoch,
                    "checkpoint_dev_loss": dev_loss,
                },
            )
        write_json(output / "training.json", result)
        print(json.dumps(result["history"][-1]), flush=True)
    return result


def load_prepared(path):
    path = Path(path)
    data = json.loads((path / "prepared.json").read_text())
    signature = data.pop("signature")
    if ev._digest(data) != signature:
        raise ValueError("Prepared examples changed; prepare a new experiment")
    data["signature"] = signature
    return data


def prepare(args):
    from transformers import AutoConfig, AutoTokenizer

    # Check existence before calling the legacy create-or-load helper.
    if not args.splits.is_file():
        raise FileNotFoundError(f"Existing split manifest required: {args.splits}")
    dialogues = ev.load_dialogues(args.corpus)
    manifest = ev.load_or_create_splits(dialogues, args.splits, args.seed)
    partitions, duplicate_audit = isolated_partitions(dialogues, manifest)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    revision = config._commit_hash
    if not revision:
        raise ValueError("Use a Hub model with a resolved immutable revision")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    span_tokenizer = AutoTokenizer.from_pretrained(
        ev.MODEL_ID, revision=args.span_revision, use_fast=True
    )
    data = {
        "model": args.model,
        "revision": revision,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "tokenizer_sha256": ev._digest(tokenizer.backend_tokenizer.to_str()),
        "span_model": ev.MODEL_ID,
        "span_revision": args.span_revision,
        "dataset_sha256": manifest["dataset_sha256"],
        "manifest_sha256": ev._digest(manifest),
        "duplicate_audit": duplicate_audit,
        "preparation": {},
        "examples": {},
        "versions": versions(),
    }
    for split, rows in partitions.items():
        data["examples"][split], data["preparation"][split] = prepare_partition(
            rows,
            span_tokenizer,
            tokenizer,
            args.seed,
            args.max_length,
            args.max_new_tokens,
        )
    if any(not rows for rows in data["examples"].values()):
        raise ValueError("Every partition must retain eligible examples")
    data["signature"] = ev._digest(data)
    tokenizer.save_pretrained(args.output / "tokenizer")
    write_json(args.output / "prepared.json", data)
    return {
        "signature": data["signature"],
        "duplicates": duplicate_audit,
        "preparation": data["preparation"],
    }


def gpu_environment():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Run on the host; no driver changes or fallback attempted."
        )
    free, total = torch.cuda.mem_get_info()
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise RuntimeError(
            "Unsloth requires compute capability >= 7.0 for this workflow"
        )
    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(capability),
        "free_bytes": free,
        "total_bytes": total,
        "cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "base_dtype": "float16",
        "compiled_cuda_architectures": torch.cuda.get_arch_list(),
    }


def training_options(args):
    return {
        name: getattr(args, name)
        for name in (
            "batch_size",
            "accumulation",
            "learning_rate",
            "rank",
            "alpha",
            "targets",
        )
    }


def run_gpu(args):
    # Must precede Transformers, PEFT and TRL imports in this fresh CLI process.
    environment = gpu_environment()
    from unsloth import FastLanguageModel

    # isort: split
    from transformers import AutoTokenizer, set_seed

    data = load_prepared(args.prepared)
    tokenizer = AutoTokenizer.from_pretrained(args.prepared / "tokenizer")
    if ev._digest(tokenizer.backend_tokenizer.to_str()) != data["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer changed")
    options = training_options(args)
    if args.command == "train":
        feasibility = json.loads((args.feasibility / "success.json").read_text())
        if (
            feasibility["signature"] != data["signature"]
            or feasibility["options"] != options
        ):
            raise ValueError(
                "Run feasibility with this prepared dataset and these training options first"
            )
    set_seed(data["seed"])
    metadata = {
        "signature": data["signature"],
        "model": data["model"],
        "revision": data["revision"],
        "seed": data["seed"],
        "options": options,
        "epochs": 1 if args.command == "feasibility" else args.epochs,
        "environment": environment,
        "versions": versions(),
        "example_counts": {s: len(r) for s, r in data["examples"].items()},
    }
    write_json(args.output / "config.json", metadata)
    # Feasibility is only a compatibility gate. Reload the original base and
    # initialize fresh adapters rather than continuing its short training run.
    model, _ = FastLanguageModel.from_pretrained(
        model_name=data["model"],
        revision=data["revision"],
        max_seq_length=data["max_length"],
        dtype=torch.float16,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=True,
        device_map={"": 0},
        full_finetuning=False,
        use_exact_model_name=True,
        use_gradient_checkpointing=True,
        offload_embedding=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=args.targets.split(","),
        lora_alpha=args.alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=data["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    # Standard checkpointing avoids Unsloth's optional activation offloading.
    model.config.use_cache = False
    if model.config._commit_hash != data["revision"]:
        raise ValueError("Loaded model revision differs from the prepared experiment")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(
        model, "is_loaded_in_8bit", False
    ):
        raise ValueError("Quantization is forbidden")
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise ValueError("All model parameters must remain on GPU")
    if any(b.device.type != "cuda" for b in model.buffers()):
        raise ValueError("All model buffers must remain on GPU")
    if any(
        p.dtype != torch.float16
        for n, p in model.named_parameters()
        if "lora_" not in n
    ):
        raise ValueError("Frozen base parameters must all be FP16")
    training = data["examples"]["train"]
    # Include the longest training example in the short memory feasibility test.
    if args.command == "feasibility":
        training = sorted(training, key=lambda r: len(r["input_ids"]), reverse=True)
        training = training[: args.batch_size * args.accumulation * args.steps]
    before = {
        n: p.detach().cpu().clone() for n, p in model.named_parameters() if "lora_" in n
    }
    result = fit(
        model,
        training,
        data["examples"]["dev"],
        args.output,
        data["pad_token_id"],
        epochs=1 if args.command == "feasibility" else args.epochs,
        batch_size=args.batch_size,
        accumulation=args.accumulation,
        learning_rate=args.learning_rate,
        seed=data["seed"],
        device="cuda",
        max_steps=args.steps if args.command == "feasibility" else None,
        metadata=metadata,
    )
    if not any(
        not torch.equal(before[n], p.detach().cpu())
        for n, p in model.named_parameters()
        if n in before
    ):
        raise ValueError("Optimizer steps did not change any adapter parameters")
    tokenizer.save_pretrained(args.output / "best")
    write_json(
        args.output / "success.json",
        {"signature": data["signature"], "options": options, "command": args.command},
    )
    return result


def safe_error(error):
    message = str(error)
    token = os.environ.get("HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    return re.sub(r"hf_[A-Za-z0-9]+", "[REDACTED]", message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--corpus", type=Path, default=Path("SW_EpisodeIV_VI.json"))
    p.add_argument("--splits", type=Path, default=Path("data/evaluation/splits.json"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default="main")
    p.add_argument(
        "--span-revision", default="86b5e0934494bd15c9632b12f734a8a67f723594"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path("data/assignment5/prepared"))
    for command in ("feasibility", "train"):
        p = sub.add_parser(command)
        p.add_argument(
            "--prepared", type=Path, default=Path("data/assignment5/prepared")
        )
        p.add_argument(
            "--output", type=Path, default=Path(f"data/assignment5/{command}")
        )
        p.add_argument("--batch-size", type=int, default=1)
        p.add_argument("--accumulation", type=int, default=8)
        p.add_argument("--learning-rate", type=float, default=2e-4)
        p.add_argument("--epochs", type=int, default=3)
        p.add_argument("--rank", type=int, default=4)
        p.add_argument("--alpha", type=int, default=8)
        p.add_argument("--targets", default="q_proj,v_proj")
        p.add_argument("--steps", type=int, default=2)
        p.add_argument(
            "--feasibility", type=Path, default=Path("data/assignment5/feasibility")
        )
    args = parser.parse_args(argv)
    for name in (
        "batch_size",
        "accumulation",
        "epochs",
        "rank",
        "alpha",
        "steps",
        "max_length",
        "max_new_tokens",
    ):
        if hasattr(args, name):
            ev._integer(name, getattr(args, name), 1)
    if hasattr(args, "seed"):
        ev._integer("seed", args.seed)
    if hasattr(args, "learning_rate") and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("learning-rate must be finite and positive")
    # Never mix failed or completed runs with a new run, or overwrite sources.
    if args.output.exists():
        parser.error("Output already exists; use a fresh directory")
    args.output.mkdir(parents=True)
    try:
        result = prepare(args) if args.command == "prepare" else run_gpu(args)
    except Exception as error:  # noqa: BLE001 - persist sanitized failures at CLI boundary
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": safe_error(error),
            "configuration": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "versions": versions(),
        }
        write_json(args.output / "failure.json", failure)
        print(json.dumps(failure), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
