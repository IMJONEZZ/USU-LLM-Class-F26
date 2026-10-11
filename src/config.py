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

# ---- teacher / judge ----
TEACHER_MODEL = "gemini-2.5-pro"
JUDGE_MODEL = "gemini-2.5-flash"
TEACHER_SAMPLES = 8
API_KEY_ENV = "GOOGLE_API_KEY"

# ---- templates ----
INSTRUCTION_MARKER = "### Instruction:\n"
RESPONSE_MARKER = "### Response:\n"
SECTION_LABELS = ["Restate:", "Check:", "Solve:", "Verify:", "Final answer:"]
FINAL_ANSWER_LABEL = "Final answer:"

# ---- paths ----
ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ADAPTER_DIR = ROOT / "adapters"
TEST_SET_PATH = ASSETS_DIR / "test_set.jsonl"
MISGUIDED_ANSWERS_PATH = ASSETS_DIR / "misguided_answers.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
EVAL_PATH = DATA_DIR / "eval.jsonl"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
ROUNDS_PATH = RESULTS_DIR / "rounds.json"
FINAL_TEST_PATH = RESULTS_DIR / "final_test.json"


def adapter_path(round_n):
    if isinstance(round_n, bool) or not isinstance(round_n, int):
        raise TypeError(f"round_n must be int, got {type(round_n).__name__}")
    if round_n < 0:
        raise ValueError(f"round_n must be >= 0, got {round_n}")
    return ADAPTER_DIR / f"round_{round_n}"
