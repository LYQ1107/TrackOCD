#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-phase46_formal_v1}"; STEPS="${2:-1000}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
mkdir -p "$ROOT/outputs/iclr27_phase46/logs" "$ROOT/outputs/iclr27_phase46/completion"
pids=()
for f in 0 1 2 3; do
  if [[ -f "$ROOT/outputs/iclr27_phase46/completion/${TAG}_f${f}.done" || -f "$ROOT/outputs/iclr27_phase46/completion/${TAG}_f${f}.launched" ]]; then continue; fi
  gpu=$((f+4))
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase46/train_gate.py" --fold "$f" --steps "$STEPS" --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" >"$ROOT/outputs/iclr27_phase46/logs/${TAG}_f${f}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
