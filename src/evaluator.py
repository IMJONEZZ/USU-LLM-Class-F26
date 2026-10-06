"""
Evaluator for measuring a causal language model's loss/perplexity on
held-out Star Wars dialogue (the Return of the Jedi test split), used to
compare model performance before and after fine-tuning.
"""

import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataloader import load_scenes, scenes_to_text, split_scenes_by_movie

MODEL_ID = "unsloth/Llama-3.2-1B"
MAX_LENGTH = 128
STRIDE = 32


def load_test_text(data_path="data/star_wars_script.jsonl"):
    """Load and flatten the held-out test split (Return of the Jedi) into text."""
    scenes = load_scenes(data_path)
    _, _, test_scenes = split_scenes_by_movie(scenes)
    return scenes_to_text(test_scenes)


def compute_loss(
    model, tokenizer, text, max_length=MAX_LENGTH, stride=STRIDE, device="cuda"
):
    """
    Compute average loss (and perplexity) of `model` on `text`, using a
    sliding window of `max_length` tokens with the given `stride`, so long
    documents can be evaluated in fixed-size chunks.
    """
    model.eval()
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    seq_len = input_ids.size(1)

    total_loss = 0.0
    total_chunks = 0

    with torch.no_grad():
        for i in range(0, seq_len - max_length, stride):
            chunk = input_ids[:, i : i + max_length]
            outputs = model(chunk, labels=chunk)
            total_loss += outputs.loss.item()
            total_chunks += 1

    if total_chunks == 0:
        raise ValueError("Text too short to form even one evaluation chunk.")

    avg_loss = total_loss / total_chunks
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def load_base_model(model_id=MODEL_ID, device="cuda"):  # pragma: no cover
    """Load the plain, un-fine-tuned base model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    return model, tokenizer


def load_finetuned_model(
    adapter_path, model_id=MODEL_ID, device="cuda"
):  # pragma: no cover
    """
    Load the base model, then attach the saved LoRA adapter on top of it,
    producing the fine-tuned model for evaluation.
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = (
        model.merge_and_unload()
    )  # merge LoRA weights into the base model for clean inference
    return model, tokenizer


def evaluate_model(
    model, tokenizer, data_path="data/star_wars_script.jsonl", device="cuda"
):  # pragma: no cover
    """
    Full evaluation pipeline: given an already-loaded model/tokenizer,
    compute loss/perplexity on the held-out test split.
    """
    test_text = load_test_text(data_path)
    avg_loss, perplexity = compute_loss(model, tokenizer, test_text, device=device)
    return {"loss": avg_loss, "perplexity": perplexity}


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "finetuned":
        print("Evaluating FINE-TUNED model...")
        model, tokenizer = load_finetuned_model("outputs/star_wars_lora")
    else:
        print("Evaluating BASE (pre-fine-tuning) model...")
        model, tokenizer = load_base_model()

    results = evaluate_model(model, tokenizer)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Perplexity: {results['perplexity']:.4f}")
