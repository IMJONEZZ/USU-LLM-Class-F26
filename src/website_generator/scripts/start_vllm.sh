#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
# Use native sampling when the WSL CUDA compiler toolkit is unavailable.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
exec uv run --group inference vllm serve "${VLLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --host 127.0.0.1 --port 8000 \
  --enforce-eager \
  --dtype auto --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.80 \
  --structured-outputs-config.backend xgrammar
