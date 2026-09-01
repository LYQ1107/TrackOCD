#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
mkdir -p "$ROOT/outputs/iclr27_phase48/logs"
declare -a pids=()
for fold in 0 1 2 3; do
  gpu=$((fold+4)); tag="phase48_formal"
  done="$ROOT/outputs/iclr27_phase48/completion/${tag}_f${fold}.done"
  launched="$ROOT/outputs/iclr27_phase48/completion/${tag}_f${fold}.launched"
  if [[ -e "$done" || -e "$launched" ]]; then continue; fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase48/train_correspondence.py" --fold "$fold" --steps 1000 --tag "$tag" --device cuda:0 --expected-physical-gpu "$gpu" > "$ROOT/outputs/iclr27_phase48/logs/${tag}_f${fold}.log" 2>&1 &
  pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
