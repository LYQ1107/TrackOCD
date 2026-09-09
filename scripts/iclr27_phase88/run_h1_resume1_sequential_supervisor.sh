#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase88"
CKPT="$OUT/checkpoints"
LOG="$OUT/logs"
MEMMAP="/data2/usr_for_deadline/trackocd_phase88/shared_features"
mkdir -p "$LOG"

run_resume() {
  local fold="$1"
  local source="$2"
  local tag="h1_false_merge_reset_resume1_f${fold}"
  if [[ -f "$OUT/completion/${tag}.done" ]]; then
    echo "SKIP_DONE $tag"
    return 0
  fi
  if [[ -f "$OUT/completion/${tag}.launched" ]]; then
    echo "REFUSE_ALREADY_LAUNCHED $tag" >&2
    return 73
  fi
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=1 \
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device cuda:0 --updates 30000 --tag "$tag" \
      --seed 88002 --event-tag fix2 --memmap-root "$MEMMAP" \
      --resume-checkpoint "$source" --checkpoint-interval 2000 \
      --loss-profile h1_false_merge_reset \
      >"$LOG/${tag}.log" 2>&1
}

run_resume 0 "$CKPT/h1_false_merge_reset_resume1_targeted_f0.pt"
run_resume 1 "$CKPT/h1_false_merge_reset_f1_step022000.pt"
run_resume 2 "$CKPT/h1_false_merge_reset_f2_step020000.pt"
run_resume 3 "$CKPT/c0v2_fix2_formal_f3.pt"
echo "H1_RESUME1_FORMAL_COMPLETE"
