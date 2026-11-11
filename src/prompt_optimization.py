"""Compare a Llama baseline with a DSPy-optimized arithmetic prompt on Modal."""

import json
from pathlib import Path

import modal

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_DATA_PATH = Path("data/arithmetic.jsonl")
DEFAULT_REPORT_PATH = Path("data/prompt_optimization_report.txt")
MAX_NEW_TOKENS = 256

# Set up my Modal image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "accelerate",
        "dspy>=3.5.0",
        "torch",
        "transformers>=4,<5",
    )
    .add_local_python_source("src")
)

app = modal.App("llama32-3b-dspy-prompt-optimization")


# Create a custom metric to compare the numeric parts of the answers.
def exact_match(gold, pred, trace=None):
    """Accept a generated demonstration only when its numeric answer is right."""
    import re

    # Strip everything out of the answers except digits, convert to integers, and compare using exact matching.
    gold_digits = re.sub(r"[^\d]", "", str(gold.answer))
    pred_digits = re.sub(r"[^\d]", "", str(pred.answer))
    if not gold_digits or not pred_digits:
        return False
    return int(gold_digits) == int(pred_digits)


class TransformersEngine:
    """Adapt a loaded Transformers causal LM to DSPy's engine interface."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def complete(self, request):
        """Generate one response to a DSPy request."""
        import uuid

        import torch
        from dspy.lm15 import Message, Response, Usage

        # Convert the DSPy messages into the chat format expected by Llama's tokenizer.
        messages = []
        if request.system:
            if isinstance(request.system, str):
                system_text = request.system
            else:
                system_text = "".join(
                    getattr(part, "text", str(part)) for part in request.system
                )
            messages.append({"role": "system", "content": system_text})

        role_map = {"developer": "system"}
        for message in request.messages:
            messages.append(
                {
                    "role": role_map.get(message.role, message.role),
                    "content": message.text,
                }
            )

        # Llama's chat template inserts the model-specific control tokens.
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # DSPy can change temperature during bootstrap retries.
        # At temperature zero, use greedy generation. Retry rounds can sample varied outputs.
        max_new_tokens = request.config.max_tokens or MAX_NEW_TOKENS
        temperature = request.config.temperature
        generation_options = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": temperature is not None and temperature > 0,
        }
        if generation_options["do_sample"]:
            generation_options["temperature"] = temperature
            if request.config.top_p is not None:
                generation_options["top_p"] = request.config.top_p

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_options)

        # Strip the input tokens so DSPy receives only the model's new response.
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = generated[0, prompt_length:]
        answer_text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()
        output_token_count = len(new_tokens)
        finish_reason = "length" if output_token_count >= max_new_tokens else "stop"

        return Response(
            id=str(uuid.uuid4()),
            model=request.model,
            message=Message.assistant(answer_text),
            finish_reason=finish_reason,
            usage=Usage(
                input_tokens=prompt_length,
                output_tokens=output_token_count,
                total_tokens=prompt_length + output_token_count,
            ),
        )


def format_program_prompt(program):
    """Return readable instructions, signature, and demos from a DSPy program."""
    # ChainOfThought contains one predictor.
    # Its signature holds the task instructions, and its demos show the examples BootstrapFewShot attached.
    predictor = program.predictors()[0]
    lines = [
        f"Instructions: {predictor.signature.instructions or '(none)'}",
        f"Signature: {predictor.signature}",
        f"Demonstrations ({len(predictor.demos)}):",
    ]

    for index, demo in enumerate(predictor.demos, start=1):
        lines.append(f"  Demo {index}:")
        for field, value in demo.items():
            if field != "augmented":
                lines.append(f"    {field}: {value}")

    return "\n".join(lines)


def format_comparison(question, baseline_prediction, optimized_prediction):
    """Show how the same question was answered before and after compilation."""
    return "\n".join(
        [
            "Prompt comparison",
            "=================",
            f"Question: {question}",
            f"Before optimization: {baseline_prediction.answer}",
            f"After optimization:  {optimized_prediction.answer}",
        ]
    )


@app.function(
    image=image,
    gpu="T4",
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def optimize_and_compare(records: list[dict[str, str]], question: str):
    """Optimize on the labeled dataset, then compare one prompt on a Modal T4."""
    import os

    import dspy
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "The Modal 'huggingface' secret must contain the HF_TOKEN variable."
        )

    # Load one copy of the model in the remote function so both runs use the same weights and hardware.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
        token=hf_token,
    )
    model.eval()

    # TransformersEngine adapts this local model to DSPy's LM interface.
    language_model = dspy.LM(
        model=MODEL_ID,
        engine=TransformersEngine(model, tokenizer),
        temperature=0,
        max_tokens=MAX_NEW_TOKENS,
    )
    dspy.configure(lm=language_model)

    # DSPy Examples mark question as input, answer remains the training target.
    examples = [
        dspy.Example(question=row["question"], answer=row["answer"]).with_inputs(
            "question"
        )
        for row in records
    ]
    # Save a baseline answer first. The same question is run again after compile.
    baseline_program = dspy.ChainOfThought("question -> answer")
    baseline_prediction = baseline_program(question=question)

    # BootstrapFewShot tries model-generated answers, keeps only examples that
    # pass exact_match, and combines those with labeled training examples.
    optimizer = dspy.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=8,
        max_rounds=5,
    )
    # All JSONL rows are used as the labeled training set for prompt demos.
    optimized_program = optimizer.compile(baseline_program, trainset=examples)
    optimized_prediction = optimized_program(question=question)

    report = [
        f"Model: {MODEL_ID}",
        f"Training examples: {len(examples)}",
        "",
        "BASELINE PROGRAM",
        "----------------",
        format_program_prompt(baseline_program),
        "",
        "OPTIMIZED PROGRAM",
        "-----------------",
        format_program_prompt(optimized_program),
        "",
        format_comparison(question, baseline_prediction, optimized_prediction),
    ]
    return "\n".join(report)


@app.local_entrypoint()
def main(
    question: str,
    data_path: str = str(DEFAULT_DATA_PATH),
    report_path: str = str(DEFAULT_REPORT_PATH),
) -> None:
    """Load training rows and run the Modal comparison on one question."""
    # Dataset parsing is done locally, Modal only sees the compact rows and query.
    dataset_path = Path(data_path)
    with dataset_path.open(encoding="utf-8") as dataset_file:
        records = [json.loads(line) for line in dataset_file if line.strip()]

    # Save the returned report locally to extract results for the assignment
    report = optimize_and_compare.remote(records, question)
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved prompt details and comparison to {output_path}")
