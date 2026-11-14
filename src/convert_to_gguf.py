"""
Merge the fine-tuned LoRA adapter into the base model (using plain peft,
not Unsloth's GGUF export, which proved unreliable in a non-interactive
Docker environment), then convert to GGUF using llama.cpp's own scripts
directly.
"""

import os
import subprocess

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "outputs/trick_question_lora"
MERGED_DIR = "/tmp/merged_model"
GGUF_OUTPUT_DIR = "outputs/trick_question_gguf"
LLAMA_CPP_DIR = "/opt/llama.cpp"

if __name__ == "__main__":  # pragma: no cover
    print("Loading and merging adapter...")
    # Load the tokenizer from the base model rather than the adapter path,
    # since the adapter's saved tokenizer config references a custom class
    # ("TokenizersBackend") specific to Unsloth's patched environment, which
    # isn't registered when Unsloth hasn't been imported. LoRA training
    # doesn't change tokenization, so the base model's tokenizer is identical.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.merge_and_unload()

    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to {MERGED_DIR}")

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    print("Converting to GGUF (f16)...")
    f16_path = f"{GGUF_OUTPUT_DIR}/model-f16.gguf"
    subprocess.run(
        [
            "python3",
            f"{LLAMA_CPP_DIR}/convert_hf_to_gguf.py",
            MERGED_DIR,
            "--outfile",
            f16_path,
            "--outtype",
            "f16",
        ],
        check=True,
    )

    print("Quantizing to q4_k_m...")
    quantized_path = f"{GGUF_OUTPUT_DIR}/model-q4_k_m.gguf"
    subprocess.run(
        [
            f"{LLAMA_CPP_DIR}/build/bin/llama-quantize",
            f16_path,
            quantized_path,
            "q4_k_m",
        ],
        check=True,
    )

    print(f"Done. Quantized model at {quantized_path}")
