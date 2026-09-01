#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/MOTIP2/bin/python"
TAG="${1:-formal}"
STEPS="${2:-1000}"
OUT="$ROOT/outputs/iclr27_phase60"
mkdir -p "$OUT/audit" "$OUT/logs" "$OUT/completion"
{
  echo "phase=60 tag=$TAG steps=$STEPS"
  date -Is
  free -h
  pgrep -af 'train_pixel_e2e|run_four_fold_supervisor' || true
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  df -h "$ROOT"
} > "$OUT/audit/resource_${TAG}_preflight.txt"

# GPU 4--7 are selected only after the preflight above confirms their state.
GPUS=(4 5 6 7)
PIDS=()
for fold in 0 1 2 3; do
  done_marker="$OUT/completion/phase60_${TAG}_f${fold}.done"
  launched_marker="$OUT/completion/phase60_${TAG}_f${fold}.launched"
  if [[ -e "$done_marker" || -e "$launched_marker" ]]; then
    echo "skip fold=$fold (existing marker)"
    continue
  fi
  tmp="$launched_marker.tmp.$$"
  printf '{"phase":60,"fold":%d,"gpu":%d,"tag":"%s","steps":%d}\n' "$fold" "${GPUS[$fold]}" "$TAG" "$STEPS" > "$tmp"
  mv "$tmp" "$launched_marker"
  CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PY" "$ROOT/scripts/iclr27_phase60/train_pixel_e2e.py" \
    --fold "$fold" --device cuda:0 --steps "$STEPS" --batch-size 4 --workers 2 \
    --seed $((575700+fold)) --tag "$TAG" --ckpt-every 100 --log-every 100 \
    > "$OUT/logs/${TAG}_f${fold}.log" 2>&1 &
  PIDS+=("$!")
  echo "launched fold=$fold gpu=${GPUS[$fold]} pid=${PIDS[-1]}"
done

if [[ ${#PIDS[@]} -gt 0 ]]; then
  # The supervisor performs exactly one blocking wait for this bounded batch.
  status=0
  for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
  exit "$status"
fi
echo "no unfinished fold to launch"
