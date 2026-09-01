#!/usr/bin/env bash
set -euo pipefail

# One bounded worker per fold. CUDA_VISIBLE_DEVICES is intentionally mapped
# to physical GPUs 4--7; each worker sees its assigned card as cuda:0.
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
OUT="$ROOT/outputs/iclr27_phase27"
mkdir -p "$OUT/logs" "$OUT/completion" "$OUT/checkpoints" "$OUT/metrics"

declare -a PIDS=()
declare -a FOLDS=()
for fold in 0 1 2 3; do
  run="correspondence_f${fold}"
  done="$OUT/completion/${run}.done"
  launched="$OUT/completion/${run}.launched"
  if [[ -e "$done" ]]; then
    echo "fold=${fold} skip=done"
    continue
  fi
  if [[ -e "$launched" ]]; then
    echo "fold=${fold} skip=launched marker=${launched}"
    continue
  fi
  gpu=$((fold + 4))
  log="$OUT/logs/${run}.stdout.log"
  echo "fold=${fold} gpu=${gpu} launch_log=${log}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase27/train_correspondence.py" \
    --fold "$fold" --device cuda:0 --expected-physical-gpu "$gpu" \
    --steps 2000 --batch-size 16 --checkpoint-every 500 --tag correspondence \
    >"$log" 2>&1 &
  PIDS+=("$!")
  FOLDS+=("$fold")
done

status=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"; fold="${FOLDS[$i]}"
  if wait "$pid"; then
    echo "fold=${fold} pid=${pid} exit=0"
  else
    rc=$?
    echo "fold=${fold} pid=${pid} exit=${rc}" >&2
    status=1
  fi
done
exit "$status"
