"""Compare unquantized models, or keep one loaded for GPU evidence."""

import argparse
import json
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-360M-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def largest_success(results):
    successes = [r for r in results if r["status"] == "success" and r.get("parameters")]
    if not successes:
        raise ValueError("No successful models with parameter counts in this sweep.")
    return max(successes, key=lambda r: r["parameters"])["model_id"]


def worker(args):
    # The parent never imports torch or owns CUDA memory.
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

    result_path = Path(args.worker)
    result = {
        "model_id": args.model,
        "parameters": None,
        "status": "error",
        "dtype": "bfloat16",
        "quantization": "none",
        "device_map": {"": 0},
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "seed": 42,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    stage = "setup"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Start Docker with --gpus all.")
        result.update(
            gpu=torch.cuda.get_device_name(0),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )
        print(f"GPU: {result['gpu']}; model: {args.model}; BF16, no quantization")
        stage = "parameter_count"
        config = AutoConfig.from_pretrained(args.model)
        # Count parameters without allocating weights, including for failed loads.
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(config)
        skeleton.tie_weights()
        result["parameters"] = skeleton.num_parameters()
        del skeleton
        print(f"Parameters: {result['parameters']:,}")
        save_json(result_path, result)
        stage = "loading"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        result["parameters"] = model.num_parameters()
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        stage = "generation"
        torch.manual_seed(42)
        output = generator(
            [{"role": "user", "content": args.prompt}],
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        result.update(
            status="success",
            generated_text=output[0]["generated_text"][-1]["content"],
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
        print("\nGenerated output:\n" + result["generated_text"])
        result_path.with_suffix(".output.txt").write_text(
            result["generated_text"] + "\n", encoding="utf-8"
        )
        # Capture while the model still holds its CUDA tensors.
        try:
            smi = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True
            ).stdout
            result_path.with_suffix(".nvidia-smi.txt").write_text(smi, encoding="utf-8")
            print("\n" + smi)
        except (OSError, subprocess.CalledProcessError) as exc:
            result["nvidia_smi_error"] = str(exc)
            print(f"nvidia-smi capture failed: {exc}")
    except Exception as exc:  # noqa: BLE001 -- Record any model failure and continue the sweep.
        result.update(
            status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
            failure_stage=stage,
            error=f"{type(exc).__name__}: {exc}",
        )
        traceback.print_exc()
    save_json(result_path, result)
    if args.hold and result["status"] == "success":
        print(
            "Model is loaded. Take your screenshot, then press Enter to exit.",
            flush=True,
        )
        try:
            input()
        except EOFError:
            print("No interactive input; use docker run -it to keep the model loaded.")
    return 0 if result["status"] == "success" else 1


def attempt(model_id, args, directory, index, hold=False):
    result_path = directory / f"{index:02d}.json"
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        str(result_path),
        "--model",
        model_id,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if hold:
        command.append("--hold")
    with (
        result_path.with_suffix(".log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process,
    ):
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result = {"model_id": model_id, "parameters": None, "status": "error"}
    if returncode and result["status"] not in ("oom", "error"):
        result["status"] = "error"
    if returncode:
        result.setdefault("error", f"Worker exited with code {returncode}; see log.")
    result["exit_code"] = returncode
    save_json(result_path, result)
    return result


def report(results):
    lines = [
        "| Model ID | Parameters (billions) | Outcome | Details |",
        "| --- | ---: | --- | --- |",
    ]
    for result in results:
        count = result.get("parameters")
        size = f"{count / 1e9:.3f}" if count is not None else "unknown"
        detail = result.get("error", "Generated successfully")
        detail = " ".join(detail.split()).replace("|", "/")
        lines.append(
            f"| {result['model_id']} | {size} | {result['status']} | {detail} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["run", "sweep"], default="run")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--model", default=None)
    choice.add_argument(
        "--largest-from", type=Path, help="Path to a sweep results.json"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=Path("/results"))
    parser.add_argument(
        "--prompt", default="Write a short story that begins: Once upon a time"
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--no-hold", action="store_true", help="Exit after a single run"
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--hold", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.worker:
        return worker(args)
    if args.mode == "sweep" and (args.model or args.largest_from):
        parser.error("Use --models for sweep mode")
    models = args.models
    if args.mode == "run":
        model_id = args.model or DEFAULT_MODELS[0]
        if args.largest_from:
            try:
                model_id = largest_success(
                    json.loads(args.largest_from.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        models = [model_id]
    directory = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Results directory: {directory}", flush=True)
    results = []
    for index, model_id in enumerate(models, start=1):
        results.append(
            attempt(
                model_id,
                args,
                directory,
                index,
                hold=args.mode == "run" and not args.no_hold,
            )
        )
        # Preserve completed attempts if the sweep is interrupted later.
        save_json(directory / "results.json", results)
        (directory / "summary.md").write_text(report(results), encoding="utf-8")
    print("\n" + report(results))
    print(f"Saved results to {directory}")
    return 0 if args.mode == "sweep" or results[0]["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
