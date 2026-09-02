#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG_DIR="$ROOT/outputs/iclr27_phase80a/logs"
mkdir -p "$LOG_DIR"
export PYTHONPATH="$ROOT"

declare -a pids=()
declare -a shards=(0 1 2 3)
declare -a gpus=(4 5 6 7)

for i in "${!shards[@]}"; do
  shard="${shards[$i]}"
  done_marker="$ROOT/outputs/iclr27_phase80a/completion/dense_shard_${shard}.done"
  launched_marker="$ROOT/outputs/iclr27_phase80a/completion/dense_shard_${shard}.launched"
  if [[ -f "$done_marker" ]]; then
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "refusing to relaunch already-launched shard $shard" >&2
    exit 3
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$i]}" "$PY" "$ROOT/scripts/iclr27_phase80a/extract_dense.py" \
    --shard "$shard" --num-shards 4 --device cuda:0 \
    >"$LOG_DIR/dense_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
