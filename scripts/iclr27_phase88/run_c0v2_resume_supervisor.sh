#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
LOG="$ROOT/outputs/iclr27_phase88/logs"
CK="$ROOT/outputs/iclr27_phase88/checkpoints"
mkdir -p "$LOG"
for fold in 0 1 2 3; do
  init=$(ls "$CK"/c0v2_formal_f${fold}_step*.pt | sort | tail -1)
  tag="c0v2_repair1_f${fold}"
  done="$ROOT/outputs/iclr27_phase88/completion/${tag}.done"
  marker="$ROOT/outputs/iclr27_phase88/completion/${tag}.launched"
  if [[ -f "$done" ]]; then echo "skip_done $tag"; continue; fi
  if [[ -f "$marker" ]]; then echo "refusing_relaunch_launched_without_done $tag" >&2; exit 2; fi
  echo "resume tag=$tag fold=$fold gpu=5 init=$init"
  "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
    --fold "$fold" --device cuda:5 --updates 20000 --tag "$tag" \
    --checkpoint-interval 2000 --resume-checkpoint "$init" \
    >"$LOG/${tag}.log" 2>&1
  echo "completed $tag"
done
echo "resume_supervisor_complete"
