#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG_ROOT="$ROOT/outputs/iclr27_phase87/logs"
mkdir -p "$LOG_ROOT"

# One evaluator is intentional: it serializes the full causal replay and keeps
# GPU/RAM usage bounded after the four training workers have exited.
for fold in 0 1 2 3; do
  tag="c0_val_f${fold}"
  ckpt="/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/c0_formal_f${fold}.pt"
  if [[ -f "$ROOT/outputs/iclr27_phase87/completion/${tag}.done" ]]; then
    echo "SKIP_DONE $tag"
    continue
  fi
  CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/iclr27_phase87/evaluate_controller.py" --fold "$fold" --checkpoint "$ckpt" --split val --tag "$tag" --device cuda:0 >"$LOG_ROOT/${tag}.log" 2>&1
  echo "VALIDATED $tag"
done

# The 76+76 replay is diagnostic only and never selects a checkpoint.
for fold in 0 1 2 3; do
  tag="c0_diag_f${fold}"
  ckpt="/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/c0_formal_f${fold}.pt"
  if [[ -f "$ROOT/outputs/iclr27_phase87/completion/${tag}.done" ]]; then
    echo "SKIP_DONE $tag"
    continue
  fi
  CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/iclr27_phase87/evaluate_controller.py" --fold "$fold" --checkpoint "$ckpt" --split held --tag "$tag" --device cuda:0 --retain-records >"$LOG_ROOT/${tag}.log" 2>&1
  echo "DIAGNOSTIC_REPLAYED $tag"
done
