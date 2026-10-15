#!/usr/bin/env bash
# Pack hybrid prepared data for Colab Drive upload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-$HOME/usu-a6-upload/a6_prepared_hybrid.tgz}"
mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" \
  data/assignment6/prepared_hybrid \
  data/assignment6/qa/hybrid_train.jsonl \
  data/assignment6/qa/hybrid_dev.jsonl
ls -lh "$OUT"
echo "Upload to Google Drive as: MyDrive/usu-a6/a6_prepared_hybrid.tgz"
echo "Then open notebooks/colab_a100_hybrid_train.ipynb on Colab (A100)."
