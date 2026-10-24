# ---- data ----
QUESTION_COUNT = 10  # a silent drop would shrink the extra-credit score

# ---- model ----
MODEL_NAME = "unsloth/Llama-3.2-1B"

# ---- serving ----
GPU_TYPE = "L4"
EVAL_MAX_SEQ_LENGTH = 2048
SERVED_BASE_NAME = "llama-3.2-1b-base"
HF_CACHE_VOLUME = "serving-hf-cache"
VLLM_CACHE_VOLUME = "serving-vllm-cache"
SCALEDOWN_WINDOW = 60  # seconds idle before the container exits

# ---- gemini (carried over from HW6) ----
API_KEY_ENV = "GOOGLE_API_KEY"
TEACHER_MAX_OUTPUT_TOKENS = 2048
TEACHER_MAX_RETRIES = 2
TEACHER_RETRY_SECONDS = 2.0
TEACHER_MAX_WAIT_SECONDS = 90
TEACHER_TIMEOUT_MS = 60_000
