import os

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
SCHEMA_INSTRUCTION = "Reply with only a JSON object that matches this JSON schema:"
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
# the benchmark runs on L4 alone; `make smoke` sets GPUS=L4,A10 so the informal smoke test can fall back
_gpus = os.environ.get("GPUS", "L4").split(",")
GPU_TYPE = _gpus[0] if len(_gpus) == 1 else _gpus
HF_CACHE_VOLUME = "serving-hf-cache"
SCALEDOWN_WINDOW = 1200  # seconds idle before the container exits; 20 min is Modal's documented max (modal.com/docs/guide/cold-start)
