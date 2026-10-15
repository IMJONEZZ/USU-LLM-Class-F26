#!/usr/bin/env bash
# Start or resume hybrid train inside this shell (intended for tmux window).
set -euo pipefail
cd /home/tshahzad/projects/USU-LLM-Class-F26
if pgrep -f 'src.trainer train --prepared data/assignment6/prepared_hybrid' >/dev/null; then
  echo "Train already running; attaching to log..."
  exec tail -n 50 -f data/assignment6/train_hybrid.log
fi
if [[ -f data/assignment6/train_hybrid/success.json ]]; then
  echo "Train already succeeded."
  exit 0
fi
# Fresh output dir required by trainer
if [[ -d data/assignment6/train_hybrid && ! -f data/assignment6/train_hybrid/success.json ]]; then
  echo "Incomplete train_hybrid/ present — move it aside before restarting."
  echo "Example: mv data/assignment6/train_hybrid data/assignment6/train_hybrid_incomplete_\$(date +%s)"
  exit 1
fi
exec .venv-training/bin/python -m src.trainer train \
  --prepared data/assignment6/prepared_hybrid \
  --feasibility data/assignment6/feasibility_hybrid \
  --output data/assignment6/train_hybrid \
  2>&1 | tee -a data/assignment6/train_hybrid.log
