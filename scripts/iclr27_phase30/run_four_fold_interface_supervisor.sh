#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-/home/lwr/anaconda3/envs/locatemot/bin/python}"
TAG="${1:-interface_formal}"
STEPS="${STEPS:-2000}"
mkdir -p "$ROOT/outputs/iclr27_phase30/logs" "$ROOT/outputs/iclr27_phase30/completion"
declare -a pids=()
for fold in 0 1 2 3; do
  gpu=$((fold+4))
  donef="$ROOT/outputs/iclr27_phase30/completion/${TAG}_f${fold}.done"
  launch="$ROOT/outputs/iclr27_phase30/completion/${TAG}_f${fold}.launched"
  if [[ -e "$donef" ]]; then echo "fold${fold}: done, skip"; continue; fi
  if [[ -e "$launch" ]]; then echo "fold${fold}: launched marker exists, refusing relaunch" >&2; exit 2; fi
  log="$ROOT/outputs/iclr27_phase30/logs/${TAG}_f${fold}.stdout.log"
  echo "launch fold=${fold} physical_gpu=${gpu} steps=${STEPS}"
  ( cd "$ROOT"; export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=.; exec "$PY" scripts/iclr27_phase30/train_support_set.py --fold "$fold" --device cuda:0 --expected-physical-gpu "$gpu" --steps "$STEPS" --checkpoint-every 500 --tag "$TAG" ) >"$log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
exit "$status"
