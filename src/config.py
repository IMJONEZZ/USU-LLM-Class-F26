# ---- model ----
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"

# ---- generation (shared by both methods) ----
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.0
BLOCK_CAP = (
    5  # most headings + paragraphs Guidance's one_or_more may emit; assumed, not tuned
)

# ---- prompts ----
BASE_PROMPT_TEMPLATE = (
    "Write a simple web page. Use only a title, headings and paragraphs.\n"
    "Request: {request}"
)
PROMPTS = [
    "A page introducing a neighborhood bakery and its morning specials.",
    "A page explaining how to care for a houseplant through the seasons.",
    "A page about the history of the bicycle.",
    "A page welcoming visitors to a small mountain hiking club.",
    "A page describing a beginner's weekly running plan.",
    "A page about how volcanoes erupt and shape the land.",
    "A page for a local library announcing its summer reading program.",
    "A page that explains what a solar eclipse is and when to watch one safely.",
    "A page about the benefits of learning a musical instrument as an adult.",
    "A page introducing a family-run farm and the seasons of its harvest.",
]

# ---- serving ----
GPU_TYPE = "L4"
EVAL_MAX_SEQ_LENGTH = 2048
SERVED_BASE_NAME = "llama-3.2-1b-base"
HF_CACHE_VOLUME = "serving-hf-cache"
VLLM_CACHE_VOLUME = "serving-vllm-cache"
SCALEDOWN_WINDOW = 1200  # seconds idle before the container exits; assumed Modal's max
