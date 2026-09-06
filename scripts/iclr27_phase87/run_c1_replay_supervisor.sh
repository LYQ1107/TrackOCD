#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG_ROOT="$ROOT/outputs/iclr27_phase87/logs"
mkdir -p "$LOG_ROOT"
for fold in 0 1 2 3; do
  ckpt="/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/c1_formal_f${fold}.pt"
  tag="c1_val_f${fold}"
  if [[ ! -f "$ROOT/outputs/iclr27_phase87/completion/${tag}.done" ]]; then
    CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/iclr27_phase87/evaluate_controller.py" --fold "$fold" --checkpoint "$ckpt" --split val --tag "$tag" --device cuda:0 --support-mode >"$LOG_ROOT/${tag}.log" 2>&1
    echo "VALIDATED $tag"
  fi
done
for fold in 0 1 2 3; do
  ckpt="/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/c1_formal_f${fold}.pt"
  tag="c1_diag_f${fold}"
  if [[ ! -f "$ROOT/outputs/iclr27_phase87/completion/${tag}.done" ]]; then
    CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/iclr27_phase87/evaluate_controller.py" --fold "$fold" --checkpoint "$ckpt" --split held --tag "$tag" --device cuda:0 --support-mode --retain-records >"$LOG_ROOT/${tag}.log" 2>&1
    echo "DIAGNOSTIC_REPLAYED $tag"
  fi
done
