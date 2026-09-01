#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG="$ROOT/outputs/iclr27_phase23/logs"
mkdir -p "$LOG"
declare -a PIDS=()
for fold in 0 1 2 3; do
  gpu=$fold
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase23/train_quality_ranker.py" \
    --fold "$fold" --device cuda:0 --steps 4000 --batch-size 256 --checkpoint-every 500 --tag ordered \
    >"$LOG/ranker_ordered_f${fold}.stdout.log" 2>"$LOG/ranker_ordered_f${fold}.stderr.log" &
  PIDS+=("$!")
done
printf '%s\n' "${PIDS[@]}" >"$LOG/ranker_ordered_supervisor_pids.txt"
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
