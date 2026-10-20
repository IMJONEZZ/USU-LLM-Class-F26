"""Evaluate generated dialogue continuations against Episode VI text."""

from src.dataloader import DATASET_NAME, TEST_FILM, combine_consecutive_lines

MODEL_NAME = "meta-llama/Llama-3.2-1B"


def make_continuation_examples(
    texts: list[str], tokenizer
) -> tuple[list[str], list[str]]:
    """Split each dialogue into a prompt and a held-out continuation."""
    prompts: list[str] = []
    references: list[str] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 4:
            continue
        split_at = len(token_ids) // 2
        prompts.append(tokenizer.decode(token_ids[:split_at], skip_special_tokens=True))
        references.append(
            tokenizer.decode(token_ids[split_at:], skip_special_tokens=True)
        )
    return prompts, references


def run_evaluation(
    model_name: str,
    adapter_path: str | None = None,
    split: str = "test",
    dataset_name: str = DATASET_NAME,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> dict[str, float]:
    """Generate continuations for Episode VI and return ROUGE scores."""
    if split != "test":
        raise ValueError("The evaluator uses the held-out Episode VI test data")

    import evaluate
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    source = load_dataset(dataset_name, "cues", split="train")
    test_rows = (row for row in source if row["film"] == TEST_FILM)
    grouped = combine_consecutive_lines(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts, references = make_continuation_examples(
        [row["text"] for row in grouped], tokenizer
    )
    if not prompts:
        raise ValueError("Episode VI has no dialogue long enough to evaluate")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    generated = generator(
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    predictions = [
        item[0]["generated_text"] if isinstance(item, list) else item["generated_text"]
        for item in generated
    ]
    return evaluate.load("rouge").compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
    )


if __name__ == "__main__":  # pragma: no cover
    print(run_evaluation(model_name=MODEL_NAME))
