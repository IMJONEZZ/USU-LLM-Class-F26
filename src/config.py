# ---- models ----
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
GEMINI_MODEL = "gemini/gemini-3.8-flash"  # writes the baseline and reflects for GEPA

# ---- serving ----
GPU_TYPE = "T4"
HF_CACHE_VOLUME = "serving-hf-cache"
SCALEDOWN_WINDOW = 1200  # seconds idle before the container exits; an idle stretch costs at most about $0.20

SERVED_NAME = "llama-3.2-3b"  # the model name clients send to the server
DTYPE = "half"  # the T4 has no bf16, so fp16
MAX_NEW_TOKENS = 2048  # the longest ask is 1200 words, about 1,600 tokens (1.3 tokens per word, assumed)
MAX_MODEL_LEN = (
    4096  # longest prompt 361 tokens + DSPy template + MAX_NEW_TOKENS, rounded up
)

# ---- data ----
SEED = 42
MIN_TYPE_PROMPTS = 6  # types with this many prompts or fewer are dropped
TRAIN_FRACTION = 0.45
VALIDATION_FRACTION = 0.30
TEST_FRACTION = 0.25

# ---- sampling ----
TEMPERATURE = 1.0  # pilot and test draws; the greedy pass uses 0
PILOT_SAMPLES = 5  # samples per prompt in the M4 pilot
TEST_DRAWS = 20  # draws per query in the M8 test run

# ---- prices (for the call and cost estimate) ----
GEMINI_INPUT_PER_MILLION = (
    0.75  # dollars; aistudio.google.com/docs/pricing, through 2026-12-31
)
GEMINI_OUTPUT_PER_MILLION = 3.75
T4_PER_SECOND = 0.000164  # dollars; modal.com/pricing

# ---- paths ----
SPLITS_PATH = "assets/splits.json"
PROMPTS_PATH = "assets/prompts.json"
RESULTS_DIR = "results"
