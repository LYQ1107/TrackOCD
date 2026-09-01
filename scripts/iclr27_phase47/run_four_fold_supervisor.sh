#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/locatemot/bin/python"; TAG="${1:-phase47_formal_v1}"; STEPS="${2:-1000}"; pids=(); mkdir -p "$ROOT/outputs/iclr27_phase47/logs"
for f in 0 1 2 3; do
  if [[ -f "$ROOT/outputs/iclr27_phase47/completion/${TAG}_f${f}.done" || -f "$ROOT/outputs/iclr27_phase47/completion/${TAG}_f${f}.launched" ]]; then continue; fi
  gpu=$((f+4)); CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase47/train_correspondence.py" --fold "$f" --steps "$STEPS" --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" >"$ROOT/outputs/iclr27_phase47/logs/${TAG}_f${f}.log" 2>&1 & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; exit "$status"
