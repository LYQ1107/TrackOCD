#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TAG="${1:-phase80b_formal}"
STEPS="${2:-5000}"
OUT="$ROOT/outputs/iclr27_phase80b"
LOG="$OUT/logs"
mkdir -p "$LOG"
declare -a pids=()
for fold in 0 1 2 3; do
  if [[ -f "$OUT/completion/${TAG}_f${fold}.done" ]]; then continue; fi
  if [[ -f "$OUT/completion/${TAG}_f${fold}.launched" ]]; then
    echo "refusing duplicate launched unit fold=$fold" >&2; exit 2
  fi
  gpu=$((fold+4))
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" \
    /home/lwr/anaconda3/envs/ovtr/bin/python "$ROOT/scripts/iclr27_phase80b/train_memory_fold.py" \
      --fold "$fold" --steps "$STEPS" --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" \
      >"$LOG/${TAG}_f${fold}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"

