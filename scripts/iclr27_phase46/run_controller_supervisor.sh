#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/locatemot/bin/python"; pids=(); mkdir -p "$ROOT/outputs/iclr27_phase46/logs"
for f in 0 1 2 3; do
  if [[ -f "$ROOT/outputs/iclr27_phase46/completion/controller_f${f}.done" || -f "$ROOT/outputs/iclr27_phase46/completion/controller_f${f}.launched" ]]; then continue; fi
  touch "$ROOT/outputs/iclr27_phase46/completion/controller_f${f}.launched"
  CUDA_VISIBLE_DEVICES=$((f+4)) PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase46/evaluate_controller.py" --fold "$f" --tag phase46_c2_v1 >"$ROOT/outputs/iclr27_phase46/logs/controller_f${f}.log" 2>&1 & pids+=("$!")
done
status=0; for pid in "${pids[@]}"; do wait "$pid" || status=1; done; exit "$status"
