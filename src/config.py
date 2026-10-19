# ---- model ----
MODEL_NAME = "unsloth/Llama-3.2-1B"
LORA_R = 16  # the rank the A6 adapter was trained with; the server's --max-lora-rank must match

# ---- templates (the A6 training prompt; the served model must see exactly this) ----
INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "### Response:\n"
SECTION_LABELS = ["Restate:", "Check:", "Solve:", "Verify:", "Final answer:"]
FINAL_ANSWER_LABEL = SECTION_LABELS[-1]
EOS_TOKEN = "<|end_of_text|>"

# ---- serving ----
GPU_TYPE = "L4"
EVAL_MAX_NEW_TOKENS = 384
EVAL_MAX_SEQ_LENGTH = 2048
SERVED_BASE_NAME = "llama-3.2-1b-base"
ADAPTER_NAME = "trick-round0"
ADAPTER_PATH = "/adapters/round_0"
ADAPTERS_VOLUME = "serving-adapters"
HF_CACHE_VOLUME = "serving-hf-cache"
VLLM_CACHE_VOLUME = "serving-vllm-cache"
SCALEDOWN_WINDOW = 60  # seconds idle before the container exits
# the client retries a 503 (server still starting) every interval, up to the timeout
RETRY_TIMEOUT_SECONDS = 300
RETRY_INTERVAL_SECONDS = 15
RUNS_DIR = "serving_runs"  # gitignored; run records are written here as JSON
