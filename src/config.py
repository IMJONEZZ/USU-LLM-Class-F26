from pathlib import Path

# ---- model ----
MODEL_NAME = "unsloth/Llama-3.2-1B"
MAX_SEQ_LENGTH = 2048  # placeholder; recomputed in M6

# ---- LoRA / training ----
LORA_R = 16
LORA_ALPHA = 16
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LEARNING_RATE = 2e-4
EPOCHS = 3
MAX_ROUNDS = 3

# ---- data ----
CATEGORIES = [
    "false_premise",
    "misconception",
    "pun_ambiguity",
    "lateral_thinking",
    "time_sensitive",
]
REFERENCE_YEAR = 2026
EXCLUDED_TEST_IDS = frozenset()
NEAR_DUPLICATE_RATIO = 0.9
EVAL_VARIANTS_PER_QUESTION = 3
EVAL_OVERSAMPLE = 3
PARAPHRASE_TEMPERATURE = 0.7
MIN_QUESTION_WORDS = 4

# ---- teacher / judge ----
TEACHER_MODEL = "gemini-3.5-flash-lite"
JUDGE_MODEL = "gemini-3.5-flash-lite"
TEACHER_TEMPERATURE = 1.0
TEACHER_MAX_OUTPUT_TOKENS = 2048
TEACHER_MAX_RETRIES = 2
TEACHER_RETRY_SECONDS = 2.0
TEACHER_MAX_WAIT_SECONDS = 90
TEACHER_ATTEMPTS = 3
TEACHER_TIMEOUT_MS = 60_000
JUDGE_TEMPERATURE = 0.0
MIN_CHAIN_CHARS = 120
MAX_CHAIN_CHARS = 1500
LONG_KEY_CHARS = 60
API_KEY_ENV = "GOOGLE_API_KEY"

# ---- templates ----
INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "### Response:\n"
SECTION_LABELS = ["Restate:", "Check:", "Solve:", "Verify:", "Final answer:"]
FINAL_ANSWER_LABEL = "Final answer:"
EOS_TOKEN = "<|end_of_text|>"

# ---- hardware (Modal prices per second from modal.com/pricing, 2026-10-11;
# memory sizes are the cards' NVIDIA specs; the Modal request strings are not
# verified yet and are confirmed when scripts/modal_app.py is written) ----
GPU_OPTIONS = {
    "T4": {"vram_gib": 16, "usd_per_second": 0.000164},
    "L4": {"vram_gib": 24, "usd_per_second": 0.000222},
    "A10": {"vram_gib": 24, "usd_per_second": 0.000306},
    "L40S": {"vram_gib": 48, "usd_per_second": 0.000542},
    "A100-40GB": {"vram_gib": 40, "usd_per_second": 0.000583},
    "A100-80GB": {"vram_gib": 80, "usd_per_second": 0.000694},
    "H100": {"vram_gib": 80, "usd_per_second": 0.001097},
}
GPU_USABLE_FRACTION = 0.85
GPU_TYPE = "L4"

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ADAPTER_DIR = ROOT / "adapters"
TEST_SET_PATH = ASSETS_DIR / "test_set.jsonl"
BRAINTEASER_PATH = ASSETS_DIR / "brainteaser.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
ROUNDS_PATH = RESULTS_DIR / "rounds.json"
FINAL_TEST_PATH = RESULTS_DIR / "final_test.json"
CHAIN_STORE_PATH = DATA_DIR / "chains.jsonl"
EVAL_STORE_PATH = DATA_DIR / "eval_candidates.jsonl"
TRAIN_SUMMARY_PATH = DATA_DIR / "train_summary.json"
TEST_SET_HASH_PATH = DATA_DIR / "test_set_hash.jsonl"
EVAL_REVIEW_PATH = DATA_DIR / "eval_review.jsonl"
EVAL_SUMMARY_PATH = DATA_DIR / "eval_summary.json"
TOKEN_AUDIT_PATH = DATA_DIR / "token_audit.json"
REVIEW_SAMPLE_PATH = DATA_DIR / "review_sample.json"
SEED = 0


def adapter_path(round_n):
    if isinstance(round_n, bool) or not isinstance(round_n, int):
        raise TypeError(f"round_n must be int, got {type(round_n).__name__}")
    if round_n < 0:
        raise ValueError(f"round_n must be >= 0, got {round_n}")
    return ADAPTER_DIR / f"round_{round_n}"
