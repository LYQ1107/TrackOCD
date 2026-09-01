#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/AVI/bin/python"
OUT="$ROOT/outputs/iclr27_phase19"
mkdir -p "$OUT/logs" "$OUT/checkpoints" "$OUT/metrics" "$OUT/completion"
SEED="${PHASE19_SEED:-1801}"
UPDATES="${PHASE19_UPDATES:-40000}"
BATCH="${PHASE19_BATCH:-32}"
VARIANT="${PHASE19_VARIANT:-main}"
declare -a PIDS=()
declare -a FOLDS=()
for fold in 0 1 2 3; do
  done_marker="$OUT/completion/${VARIANT}_fold${fold}.done"
  launched_marker="$OUT/completion/${VARIANT}_fold${fold}.launched"
  if [[ -f "$done_marker" ]]; then
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "refusing to relaunch unfinished unit fold=${fold}; marker=${launched_marker}" >&2
    exit 3
  fi
  tmp="${launched_marker}.tmp.$$"
  printf 'launched pid-pending fold=%s seed=%s\n' "$fold" "$SEED" > "$tmp"
  mv "$tmp" "$launched_marker"
  gpu="$fold"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m src.iclr27_phase19.training.train_rollout \
    --fold "$fold" --seed "$SEED" --updates "$UPDATES" --batch-size "$BATCH" \
    --device cuda:0 --amp bf16 --ladder L2 --allow-defer --variant "$VARIANT" \
    --best "$OUT/checkpoints/${VARIANT}_fold${fold}_best.pt" \
    --latest "$OUT/checkpoints/${VARIANT}_fold${fold}_latest.pt" \
    --summary "$OUT/metrics/${VARIANT}_fold${fold}.json" \
    --done "$done_marker" > "$OUT/logs/${VARIANT}_fold${fold}.log" 2>&1 &
  pid=$!
  printf 'pid=%s fold=%s gpu=%s\n' "$pid" "$fold" "$gpu" > "$launched_marker"
  PIDS+=("$pid"); FOLDS+=("$fold")
done
if ((${#PIDS[@]})); then
  free -h >&2
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
fi
for fold in 0 1 2 3; do
  test -f "$OUT/completion/${VARIANT}_fold${fold}.done"
  test -s "$OUT/metrics/${VARIANT}_fold${fold}.json"
done
echo "PHASE19_STAGE_COMPLETE variant=$VARIANT updates=$UPDATES seed=$SEED"
