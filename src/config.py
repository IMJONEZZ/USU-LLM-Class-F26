# ---- data ----
QUESTION_COUNT = 10
TEST_SET_PATH = "assets/test_set.jsonl"
DOCS_PATH = "assets/docs.json"
RESULTS_PATH = "results/rag.json"
CHROMA_DIR = "chroma_db"

# ---- documents ----
TEACHER_MODEL = "gemini-3.5-flash-lite"
DOC_TEMPERATURE = 0.0
MIN_DOC_CHARS = 600  # about 300 tokens is about 1200 characters
MAX_DOC_CHARS = 2400

# ---- embeddings ----
EMBED_MODEL = "gemini-embedding-001"
EMBED_FULL_DIM = 3072
DIMS = [3072, 1536, 768, 256, 128, 64, 32]

# ---- rag ----
JUDGE_MODEL = "gemini-3.5-flash-lite"
JUDGE_TEMPERATURE = 0.0
RETRY_INTERVAL_SECONDS = 15  # a 503 means the Modal container is still starting
RETRY_TIMEOUT_SECONDS = 600
ANSWER_MAX_TOKENS = 64
ANSWER_STOP = ["\n"]  # the base model has no end-of-turn habit and would keep writing

# ---- model ----
MODEL_NAME = "unsloth/Llama-3.2-1B"

# ---- serving ----
GPU_TYPE = "L4"
EVAL_MAX_SEQ_LENGTH = 2048
SERVED_BASE_NAME = "llama-3.2-1b-base"
HF_CACHE_VOLUME = "serving-hf-cache"
VLLM_CACHE_VOLUME = "serving-vllm-cache"
SCALEDOWN_WINDOW = 1200  # seconds idle before the container exits; assumed Modal's max

# ---- gemini (carried over from HW6) ----
API_KEY_ENV = "GOOGLE_API_KEY"
TEACHER_MAX_OUTPUT_TOKENS = 2048
TEACHER_MAX_RETRIES = 2
TEACHER_RETRY_SECONDS = 2.0
TEACHER_MAX_WAIT_SECONDS = 90
TEACHER_TIMEOUT_MS = 60_000
