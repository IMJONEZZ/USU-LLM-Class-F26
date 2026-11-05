__generated_with = "0.25.2"

# %%
import importlib.metadata
import json
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import dspy
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# %%
class SingleOutputAdapter(dspy.Adapter):
    """Plain text avoids asking small Qwen to emit DSPy markers or JSON.

    Both the riddle task and MIPRO's instruction proposer have one string output.
    This uses DSPy's Adapter extension point and preserves instruction/demo search.
    """

    def format(self, signature, demos, inputs):
        if len(signature.output_fields) != 1:
            raise ValueError("This experiment's adapter requires one output field.")
        output_name, output_field = next(iter(signature.output_fields.items()))
        if output_field.annotation is not str:
            raise ValueError("This experiment's adapter requires a string output.")

        def user_text(values):
            if len(signature.input_fields) == 1:
                return str(values[next(iter(signature.input_fields))])
            return "\n\n".join(
                f"{name}: {values[name]}"
                for name in signature.input_fields
                if name in values
            )

        instructions = signature.instructions
        if output_name != "answer":
            # The proposer needs its output-field description, otherwise it can
            # mistake the example riddles for questions it should answer.
            description = output_field.json_schema_extra.get("desc", output_name)
            instructions += (
                f"\n\nRequired output ({output_name}): {description}\n"
                "Return only this output as plain text."
            )
        messages = [{"role": "system", "content": instructions}]
        for demo in demos:
            if output_name in demo and all(
                name in demo for name in signature.input_fields
            ):
                messages.extend(
                    [
                        {"role": "user", "content": user_text(demo)},
                        {"role": "assistant", "content": str(demo[output_name])},
                    ]
                )
        content = user_text(inputs)
        if output_name == "proposed_instruction":
            content += (
                "\n\nWrite a new instruction for the model that will answer these "
                "riddles. Do not answer the example riddles yourself."
            )
        messages.append({"role": "user", "content": content})
        return messages

    def parse(self, signature, completion):
        return {next(iter(signature.output_fields)): completion.strip()}


# %%
class TransformersLM(dspy.BaseLM):
    """DSPy 3.3 custom backend; copies share the model, tokenizer, and lock."""

    forward_contract = "typed_lm"

    def __init__(self, model, hf_model, tokenizer, **kwargs):
        super().__init__(model=model, cache=False, **kwargs)
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.inference_lock = threading.RLock()

    def __deepcopy__(self, memo):
        copied = self.copy()
        memo[id(self)] = copied
        return copied

    def forward(self, request):
        if request.tools or request.config.response_format is not None:
            raise ValueError("This local backend supports text chat only.")
        messages = []
        for message in request.messages:
            if any(part.type != "text" for part in message.parts):
                raise ValueError("This experiment accepts text messages only.")
            messages.append({"role": message.role, "content": message.text or ""})
        temperature = request.config.temperature or 0.0
        max_tokens = request.config.max_tokens or 64
        with self.inference_lock, torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.hf_model.device)
            if inputs["input_ids"].device.type != "cuda":
                raise RuntimeError("Qwen inputs are not on CUDA.")
            input_length = inputs["input_ids"].shape[-1]
            if input_length + max_tokens > self.hf_model.config.max_position_embeddings:
                raise ValueError("Prompt exceeds Qwen's context window; reduce demos.")
            options = {
                "do_sample": temperature > 0,
                "max_new_tokens": max_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                options.update(
                    temperature=temperature, top_p=request.config.top_p or 0.9
                )
            outputs = []
            output_tokens = 0
            for _ in range(request.config.n or 1):
                generated = self.hf_model.generate(**inputs, **options)
                continuation = generated[0, input_length:]
                text = self.tokenizer.decode(continuation, skip_special_tokens=True)
                for stop in request.config.stop or []:
                    text = text.split(stop, 1)[0]
                outputs.append(dspy.LMResponse.from_text(text.strip()).output)
                output_tokens += len(continuation)
        return dspy.LMResponse(
            model=self.model,
            outputs=outputs,
            usage={"input_tokens": input_length, "output_tokens": output_tokens},
        )


# %%
def prepare_examples(dataset):
    if len(dataset) != 100:
        raise ValueError(f"Expected 100 riddles for 70/20/10, got {len(dataset)}.")
    dataset = dataset.add_column("row_id", list(range(len(dataset))))
    outer = dataset.train_test_split(test_size=0.1, seed=42)
    inner = outer["train"].train_test_split(test_size=20, seed=42)
    partitions = {
        "train": inner["train"],
        "validation": inner["test"],
        "test": outer["test"],
    }
    examples, manifest = {}, {}
    seen_questions = set()
    for name, partition in partitions.items():
        questions = {" ".join(row["input"].lower().split()) for row in partition}
        if seen_questions & questions:
            raise ValueError(
                "Duplicate riddles cross split boundaries; resolve before optimization."
            )
        seen_questions.update(questions)
        examples[name] = [
            dspy.Example(
                question=row["input"], answer=row["expected_answer"]
            ).with_inputs("question")
            for row in partition
        ]
        manifest[name] = list(partition["row_id"])
    return examples, manifest


# %%
def make_metric(embedder, examples):
    """Normalized dot product equals the previous dot/(norm*norm) cosine metric."""
    answers = list(dict.fromkeys(example.answer for example in examples))
    with torch.inference_mode():
        vectors = embedder.encode(
            answers, normalize_embeddings=True, show_progress_bar=False
        )
    expected_vectors = dict(zip(answers, vectors, strict=True))

    def cosine_metric(example, prediction, trace=None):
        if not prediction.answer.strip():
            return 0.0
        with torch.inference_mode():
            generated_vector = embedder.encode(
                prediction.answer, normalize_embeddings=True, show_progress_bar=False
            )
        return float(
            np.clip(np.dot(expected_vectors[example.answer], generated_vector), -1, 1)
        )

    return cosine_metric


# %%
def evaluate(program, examples, metric, lm, adapter):
    rows = []
    with dspy.context(lm=lm, adapter=adapter):
        for example in examples:
            prediction = program(question=example.question)
            rows.append(
                {
                    "input": example.question,
                    "expected_answer": example.answer,
                    "generated_answer": prediction.answer,
                    "similarity": metric(example, prediction),
                }
            )
    output_df = pd.DataFrame(rows)
    return float(output_df["similarity"].mean()), output_df


# %%
def describe_prompt(program):
    predictor = program.predictors()[0]
    return {
        "instructions": predictor.signature.instructions,
        "demos": [
            {"question": demo["question"], "answer": demo["answer"]}
            for demo in predictor.demos
        ],
    }


def main():  # pragma: no cover -- requires CUDA and downloaded model weights
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # %%
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # %%
    prompt_base = (
        "Solve the user's riddle. Respond with only the name of the answer "
        "in 1 to 3 words. Omit introductory text, explanations, and quotation marks."
    )

    # %%
    # Start with the baseline. Changing this cell does not reload the models.
    run_optimization = True
    optimizer_config = {
        "auto": None,
        "num_candidates": 3,
        "max_bootstrapped_demos": 1,
        "max_labeled_demos": 2,
        "num_threads": 1,
        "max_errors": 1,
        "metric_threshold": 0.75,
        "seed": 42,
        "init_temperature": 0.7,
    }
    compile_config = {
        "num_trials": 6,
        "minibatch": False,
        "program_aware_proposer": False,
        "data_aware_proposer": False,
    }
    # Skip summary calls to keep the search small; instruction/demo search remains.

    # %%
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable: torch={torch.__version__}, CUDA={torch.version.cuda}. "
            "Run uv sync with the Windows CUDA source in pyproject.toml, check "
            "nvidia-smi, and restart. This experiment requires a CUDA GPU."
        )
    cuda_device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(cuda_device)
    _probe = torch.ones(2, device=cuda_device).square().sum().item()
    torch.cuda.synchronize()
    set_seed(42)
    print(f"GPU: {gpu_name}; PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")

    # %%
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .to(cuda_device)
        .eval()
    )
    if any(p.device != cuda_device for p in hf_model.parameters()):
        raise RuntimeError("Some Qwen parameters are not on cuda:0.")
    print(f"Qwen parameters: {hf_model.device}")

    # %%
    embedder = SentenceTransformer(embedding_model_name, device=str(cuda_device))
    if any(p.device != cuda_device for p in embedder.parameters()):
        raise RuntimeError("Some embedding parameters are not on cuda:0.")
    print(f"Embedding parameters: {embedder.device}")

    # %%
    # Supported custom backend keeps Transformers in-process; no serving system.
    # copy() shares the already-loaded Qwen model, tokenizer, and inference lock.
    task_lm = TransformersLM(
        model_name, hf_model, tokenizer, temperature=0.0, max_tokens=64
    )
    proposal_lm = task_lm.copy(temperature=0.7, max_tokens=384)
    assert proposal_lm.hf_model is task_lm.hf_model
    # Both conditions use identical plain-text formatting.
    chat_adapter = SingleOutputAdapter()
    riddle_signature = dspy.Signature(
        {
            "question": dspy.InputField(),
            "answer": dspy.OutputField(
                desc="Short answer, ideally 1 to 3 words; no explanation."
            ),
        },
        instructions=prompt_base,
    )
    baseline_program = dspy.Predict(riddle_signature)

    # %%
    raw_dataset = load_dataset("csthomps/riddles", split="train")
    split_examples, split_manifest = prepare_examples(raw_dataset)
    train_examples = split_examples["train"]
    validation_examples = split_examples["validation"]
    test_examples = split_examples["test"]
    dataset_fingerprint = raw_dataset._fingerprint
    print(
        f"Train / validation / test: {len(train_examples)} / {len(validation_examples)} / {len(test_examples)}"
    )

    # %%
    # Test embeddings are kept out of the optimization metric's closure.
    optimization_metric = make_metric(embedder, train_examples + validation_examples)

    # %%
    # Short training-only checks before evaluating the baseline or compiling.
    with dspy.context(lm=task_lm, adapter=chat_adapter):
        _prediction = baseline_program(question=train_examples[0].question)
    _score = optimization_metric(train_examples[0], _prediction)
    _self_score = optimization_metric(
        train_examples[0], dspy.Prediction(answer=train_examples[0].answer)
    )
    if not np.isfinite(_score) or not np.isclose(_self_score, 1.0, atol=1e-4):
        raise RuntimeError("Cosine metric smoke test failed.")
    smoke_passed = True
    print(
        f"Smoke test answer: {_prediction.answer}; cosine: {_score:.4f}; self-similarity: {_self_score:.4f}"
    )
    print("DSPy task output parsing passed.")

    # %%
    assert smoke_passed
    test_metric = make_metric(embedder, test_examples)
    baseline_mean, baseline_results = evaluate(
        baseline_program, test_examples, test_metric, task_lm, chat_adapter
    )
    print(f"Baseline mean cosine similarity: {baseline_mean:.4f}")
    print(baseline_results.to_string(index=False))

    # %%
    optimized_program = None
    run_directory = None
    search_results = pd.DataFrame()
    if run_optimization:
        # Enforce baseline completion, without exposing test data to the optimizer.
        assert len(baseline_results) == 10
        set_seed(42)
        run_directory = Path("artifacts/prompt_optimization") / datetime.now(
            UTC
        ).strftime("%Y%m%dT%H%M%S%fZ")
        run_directory.mkdir(parents=True, exist_ok=False)
        with dspy.context(lm=task_lm, adapter=chat_adapter):
            _optimizer = dspy.MIPROv2(
                metric=optimization_metric,
                task_model=task_lm,
                prompt_model=proposal_lm,
                teacher_settings={"temperature": 0.0},
                log_dir=str(run_directory / "search"),
                **optimizer_config,
            )
            optimized_program = _optimizer.compile(
                baseline_program.deepcopy(),
                trainset=train_examples,
                valset=validation_examples,
                **compile_config,
            )
        _trials = []
        for _trial_id, _trial in optimized_program.trial_logs.items():
            _candidate = _trial.get("full_eval_program")
            if _candidate is not None:
                _details = describe_prompt(_candidate)
                _trials.append(
                    {
                        "trial": _trial_id,
                        "validation_cosine": _trial["full_eval_score"] / 100.0,
                        "instructions": _details["instructions"],
                        "demos": json.dumps(_details["demos"], ensure_ascii=False),
                    }
                )
        search_results = pd.DataFrame(_trials)
        _distinct = search_results[["instructions", "demos"]].drop_duplicates()
        if len(_distinct) < 2:
            raise RuntimeError(
                "Search evaluated no alternative prompt. Inspect search logs."
            )
        _selected = describe_prompt(optimized_program)
        _changed = _selected != describe_prompt(baseline_program)
        # Unbound programs save the signature and demos, not model weights.
        optimized_program.save(str(run_directory / "optimized_program.json"))
        _results_directory = Path("results/prompt_optimization")
        _results_directory.mkdir(parents=True, exist_ok=True)
        (_results_directory / "final_prompt.json").write_text(
            json.dumps(_selected, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        search_results.to_csv(_results_directory / "validation_trials.csv", index=False)
        _configuration = {
            "model": model_name,
            "embedding_model": embedding_model_name,
            "gpu": gpu_name,
            "seed": 42,
            "dataset": "csthomps/riddles",
            "dataset_fingerprint": dataset_fingerprint,
            "split_row_ids": split_manifest,
            "adapter": "SingleOutputAdapter",
            "task_generation": task_lm.kwargs,
            "proposal_generation": proposal_lm.kwargs,
            "optimizer": optimizer_config,
            "compile": compile_config,
            "selected_prompt_changed": _changed,
            "versions": {
                name: importlib.metadata.version(name)
                for name in [
                    "dspy",
                    "torch",
                    "transformers",
                    "sentence-transformers",
                    "datasets",
                    "marimo",
                ]
            },
        }
        (run_directory / "configuration.json").write_text(
            json.dumps(_configuration, indent=2), encoding="utf-8"
        )
        print(
            f"Distinct prompts evaluated: {len(_distinct)}; selected prompt changed: {_changed}"
        )
        print("Selected instructions:", _selected["instructions"])
        print("Selected demos:", json.dumps(_selected["demos"], indent=2))
        print(search_results.to_string(index=False))
        print(f"Saved to {run_directory}")
    else:
        print(
            "Optimization disabled. Set run_optimization = True after checking the baseline."
        )

    # %%
    comparison = pd.DataFrame()
    comparison_summary = pd.DataFrame()
    if optimized_program is not None:
        _results_directory = Path("results/prompt_optimization")
        _results_directory.mkdir(parents=True, exist_ok=True)
        _comparisons = []
        _summaries = []
        # Report training and validation performance after prompt selection.
        # Only test performance measures generalization to held-out examples.
        for _split, _examples, _metric in [
            ("train", train_examples, optimization_metric),
            ("validation", validation_examples, optimization_metric),
            ("test", test_examples, test_metric),
        ]:
            if _split == "test":
                _baseline_mean, _baseline_results = baseline_mean, baseline_results
            else:
                _baseline_mean, _baseline_results = evaluate(
                    baseline_program, _examples, _metric, task_lm, chat_adapter
                )
            _optimized_mean, _optimized_results = evaluate(
                optimized_program, _examples, _metric, task_lm, chat_adapter
            )
            _comparison = _baseline_results.rename(
                columns={
                    "generated_answer": "baseline_answer",
                    "similarity": "baseline_similarity",
                }
            ).copy()
            _comparison.insert(0, "split", _split)
            _comparison["optimized_answer"] = _optimized_results["generated_answer"]
            _comparison["optimized_similarity"] = _optimized_results["similarity"]
            _comparison["difference"] = (
                _comparison["optimized_similarity"] - _comparison["baseline_similarity"]
            )
            _summary = {
                "split": _split,
                "examples": len(_examples),
                "baseline_mean": _baseline_mean,
                "optimized_mean": _optimized_mean,
                "difference": _optimized_mean - _baseline_mean,
            }
            _comparisons.append(_comparison)
            _summaries.append(_summary)
            _comparison.to_csv(
                _results_directory / f"{_split}_comparison.csv", index=False
            )
            (run_directory / f"{_split}_summary.json").write_text(
                json.dumps(_summary, indent=2), encoding="utf-8"
            )
        comparison = pd.concat(_comparisons, ignore_index=True)
        comparison_summary = pd.DataFrame(_summaries)
        comparison.to_csv(_results_directory / "all_splits_comparison.csv", index=False)
        comparison_summary.to_csv(
            _results_directory / "metrics_by_split.csv", index=False
        )
        print(f"Result CSVs saved to {_results_directory.resolve()}")
        print(comparison_summary.to_string(index=False))
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
