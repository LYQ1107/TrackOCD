#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
LOG="$ROOT/outputs/iclr27_phase88/logs"
mkdir -p "$LOG"
declare -a pids=()
declare -a tags=()
declare -a gpus=(5 6 7 8)
for fold in 0 1 2 3; do
  tag="c0v2_formal_f${fold}"
  marker="$ROOT/outputs/iclr27_phase88/completion/${tag}.launched"
  done="$ROOT/outputs/iclr27_phase88/completion/${tag}.done"
  if [[ -f "$done" ]]; then
    echo "skip_done $tag"
    continue
  fi
  if [[ -f "$marker" ]]; then
    echo "refusing_relaunch_launched_without_done $tag" >&2
    exit 2
  fi
  gpu="${gpus[$fold]}"
  tags+=("$tag")
  "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
    --fold "$fold" --device "cuda:${gpu}" --updates 20000 --tag "$tag" \
    --checkpoint-interval 2000 >"$LOG/${tag}.log" 2>&1 &
  pids+=("$!")
  echo "launched tag=$tag fold=$fold gpu=$gpu pid=${pids[-1]}"
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
echo "supervisor_complete status=$status tags=${tags[*]}"
exit "$status"
