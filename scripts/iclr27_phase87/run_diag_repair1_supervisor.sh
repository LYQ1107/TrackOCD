#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="/home/lwr/anaconda3/envs/locatemot/bin/python"
LOG_ROOT="$ROOT/outputs/iclr27_phase87/logs"
mkdir -p "$LOG_ROOT"
for fold in 0 1 2 3; do
  tag="c0_diag_repair1_f${fold}"
  ckpt="/data2/usr_for_deadline/trackocd_phase87/project_outputs/checkpoints/c0_formal_f${fold}.pt"
  if [[ -f "$ROOT/outputs/iclr27_phase87/completion/${tag}.done" ]]; then
    echo "SKIP_DONE $tag"
    continue
  fi
  CUDA_VISIBLE_DEVICES=5 "$PYTHON" "$ROOT/scripts/iclr27_phase87/evaluate_controller.py" --fold "$fold" --checkpoint "$ckpt" --split held --tag "$tag" --device cuda:0 --retain-records >"$LOG_ROOT/${tag}.log" 2>&1
  echo "DIAGNOSTIC_REPLAYED $tag"
done
