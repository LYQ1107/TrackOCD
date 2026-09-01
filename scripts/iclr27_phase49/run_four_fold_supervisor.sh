#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY=/home/lwr/anaconda3/envs/locatemot/bin/python
mkdir -p "$ROOT/outputs/iclr27_phase49/logs"
TAG="${TAG:-phase49_formal}"
pids=()
for fold in 0 1 2 3; do
 gpu=$((fold+4)); run="${TAG}_f${fold}"; done="$ROOT/outputs/iclr27_phase49/completion/${run}.done"; launched="$ROOT/outputs/iclr27_phase49/completion/${run}.launched"
 if [[ -e "$done" || -e "$launched" ]]; then continue; fi
 CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase49/train_residual.py" --fold "$fold" --steps 1000 --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" > "$ROOT/outputs/iclr27_phase49/logs/${run}.log" 2>&1 & pids+=("$!")
done
for p in "${pids[@]}"; do wait "$p"; done
