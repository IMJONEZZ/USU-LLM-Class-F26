#!/usr/bin/env bash
# Start the riddle model's OpenAI-compatible HTTP API. Stop with Ctrl+C.
# Requires uv and a GPU supported by vLLM.
# Usage: ./start_vllm.sh [additional vLLM arguments]
# Example: ./start_vllm.sh --gpu-memory-utilization 0.7
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${MODEL_DIR:-${REPO_DIR}/models/Qwen2.5-1.5B-Instruct-riddles}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    printf 'Model config not found: %s/config.json\n' "${MODEL_DIR}" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/\n' >&2
    exit 1
fi

printf 'Starting riddles API at http://%s:%s/v1\n' "${HOST}" "${PORT}"
# Isolate vLLM's Python/PyTorch requirements from the training project.
# The first run downloads Python (if needed) and the server dependencies.
exec uv run --no-project --python 3.12 --with vllm \
    vllm serve "${MODEL_DIR}" \
    --served-model-name riddles \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    "$@"
