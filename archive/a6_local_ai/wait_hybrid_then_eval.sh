#!/usr/bin/env bash
# Wait for hybrid train to finish, then graded-10 eval vs public_only + baseline.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
LOG=data/assignment6/train_hybrid.log
SUCCESS=data/assignment6/train_hybrid/success.json
FAIL=data/assignment6/train_hybrid/failure.json

echo "[$(date -Is)] waiting for train pid or success.json ..."
while true; do
  if [[ -f "$SUCCESS" ]]; then
    echo "[$(date -Is)] success.json present"
    break
  fi
  if [[ -f "$FAIL" ]]; then
    echo "[$(date -Is)] FAILURE:" >&2
    cat "$FAIL" >&2
    exit 1
  fi
  if ! pgrep -f 'src.trainer train --prepared data/assignment6/prepared_hybrid' >/dev/null; then
    # process gone — give log a moment to flush
    sleep 5
    if [[ -f "$SUCCESS" ]]; then
      break
    fi
    echo "[$(date -Is)] train process exited without success.json" >&2
    tail -n 40 "$LOG" >&2 || true
    exit 1
  fi
  # progress line
  tail -n 1 "$LOG" 2>/dev/null | grep -E 'optimizer_steps|training_epoch|"epoch"' || true
  sleep 60
done

echo "[$(date -Is)] running graded eval ..."
.venv-training/bin/python data/assignment6/eval_hybrid_graded.py
echo "[$(date -Is)] done"
