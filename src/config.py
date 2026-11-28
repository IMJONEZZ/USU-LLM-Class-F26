from pathlib import Path

MODAL_URL = "https://madisonbsperry--vllm-inference-server.us-west.modal.direct"

N_GPU = 1
MINUTES = 60
VLLM_PORT = 8000
ROUTING_REGION = "us-west"
GPU = "A10"
SCALEDOWN_WINDOW = 15 * MINUTES
STARTUP_TIMEOUT = 8 * MINUTES

MODEL_NAME = "unsloth/Llama-3.2-1B"
MODEL_DIR = Path("/root/.cache/models")

VLLM_URL = "/v1/completions"
