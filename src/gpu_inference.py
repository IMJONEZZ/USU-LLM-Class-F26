import atexit
import os
import platform
import sys
import time
from datetime import UTC, datetime

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.2-1B")
PROMPT = os.getenv("PROMPT", "Once upon a time")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
PAUSE_SECONDS = int(os.getenv("PAUSE_SECONDS", "0"))
HF_TOKEN = os.getenv("HF_TOKEN")


def gib(value: int) -> float:
    return value / (1024**3)


def final_memory_report() -> None:
    """Report partial allocations even when loading or generation raises."""
    print(f"Finished (UTC): {datetime.now(UTC).isoformat()}")
    if torch.cuda.is_initialized():
        print(f"Final allocated bytes: {torch.cuda.memory_allocated()}")
        print(f"Final reserved bytes: {torch.cuda.memory_reserved()}")
        print(f"Peak allocated bytes: {torch.cuda.max_memory_allocated()}")


atexit.register(final_memory_report)

print(f"Started (UTC): {datetime.now(UTC).isoformat()}")
print(f"Python: {platform.python_version()}")
print("Provider: Local")
print("Quantization: none")
print("CPU offloading: no")
print(f"Prompt: {PROMPT}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Temperature: {TEMPERATURE}")
print(f"Pause seconds: {PAUSE_SECONDS}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this container.")

device = torch.device("cuda:0")

print("=== Environment ===")
print(f"Model ID: {MODEL_ID}")
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PyTorch CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

properties = torch.cuda.get_device_properties(0)
print(f"GPU memory reported by PyTorch: {gib(properties.total_memory):.2f} GiB")
print("Requested dtype: float16")
print()

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

print("Loading model...")
torch.cuda.reset_peak_memory_stats()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,
    token=HF_TOKEN,
)

parameter_count = sum(parameter.numel() for parameter in model.parameters())
print(f"Parameters: {parameter_count:,}")
print("CPU deserialization completed; transferring full model to cuda:0...")

# Explicitly require the full model to reside on the GPU.
model = model.to(device)
model.eval()

parameter_devices = {str(parameter.device) for parameter in model.parameters()}
buffer_devices = {str(buffer.device) for buffer in model.buffers()}

print()
print("=== Model Loaded ===")
print(f"Parameters: {parameter_count:,}")
print(f"Parameter devices: {sorted(parameter_devices)}")
print(f"Buffer devices: {sorted(buffer_devices)}")

if parameter_devices != {"cuda:0"}:
    raise RuntimeError(
        f"Model is not entirely on cuda:0. Found devices: {parameter_devices}"
    )

if buffer_devices - {"cuda:0"}:
    raise RuntimeError(f"Model has buffers outside cuda:0: {buffer_devices}")

print(f"Allocated after load bytes: {torch.cuda.memory_allocated()}")
print(f"Reserved after load bytes: {torch.cuda.memory_reserved()}")
print(f"Allocated after load: {gib(torch.cuda.memory_allocated()):.2f} GiB")
print(f"Reserved after load: {gib(torch.cuda.memory_reserved()):.2f} GiB")

if PAUSE_SECONDS > 0:
    print()
    print(
        f"Model loaded successfully. Pausing for {PAUSE_SECONDS} seconds "
        "for nvidia-smi inspection..."
    )
    sys.stdout.flush()
    time.sleep(PAUSE_SECONDS)

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

print()
print("=== Generation Settings ===")
print(f"Prompt: {PROMPT}")
print(f"Max new tokens: {MAX_NEW_TOKENS}")
print(f"Temperature: {TEMPERATURE}")

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print()
print("=== Generated Text ===")
print(generated_text)

print()
print("=== GPU Memory ===")
print(f"Allocated: {gib(torch.cuda.memory_allocated()):.2f} GiB")
print(f"Reserved: {gib(torch.cuda.memory_reserved()):.2f} GiB")
print(f"Peak allocated: {gib(torch.cuda.max_memory_allocated()):.2f} GiB")

print()
print("RESULT: SUCCESS")
