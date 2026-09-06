#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG_ROOT="$ROOT/outputs/iclr27_phase87/logs"
mkdir -p "$LOG_ROOT"

declare -a PIDS=()
declare -a FOLDS=()
for spec in "0:5" "1:6" "2:7" "3:8"; do
  fold="${spec%%:*}"
  gpu="${spec##*:}"
  tag="c0_formal_f${fold}"
  done="$ROOT/outputs/iclr27_phase87/completion/${tag}.done"
  launched="$ROOT/outputs/iclr27_phase87/completion/${tag}.launched"
  if [[ -f "$done" ]]; then
    echo "SKIP_DONE fold=$fold tag=$tag"
    continue
  fi
  if [[ -f "$launched" ]]; then
    echo "SKIP_ALREADY_LAUNCHED fold=$fold marker=$launched" >&2
    continue
  fi
  echo "LAUNCH fold=$fold gpu=$gpu tag=$tag"
  ( export CUDA_VISIBLE_DEVICES="$gpu"; exec "$PYTHON" "$ROOT/scripts/iclr27_phase87/train_controller.py" --fold "$fold" --device cuda:0 --updates 20000 --tag "$tag" >"$LOG_ROOT/${tag}.log" 2>&1 ) &
  PIDS+=("$!")
  FOLDS+=("$fold")
done

status=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "EXIT_OK fold=${FOLDS[$i]} pid=${PIDS[$i]}"
  else
    echo "EXIT_FAIL fold=${FOLDS[$i]} pid=${PIDS[$i]}" >&2
    status=1
  fi
done
exit "$status"
