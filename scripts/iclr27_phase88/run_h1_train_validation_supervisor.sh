#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase88"
MEMMAP="/data2/usr_for_deadline/trackocd_phase88/shared_features"
run_val() {
  local fold="$1"
  local ckpt="$2"
  local tag="h1_trainval_f${fold}"
  if [[ -f "$OUT/validation/$tag/final.done" ]]; then echo "SKIP_DONE $tag"; return 0; fi
  CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=1 \
    "$PY" "$ROOT/scripts/iclr27_phase88/evaluate_checkpoint_sharded.py" \
      --fold "$fold" --checkpoint "$ckpt" --tag "$tag" --event-tag fix2 \
      --device cuda:0 --shard-size 250 --memmap-root "$MEMMAP" \
      >"$OUT/logs/${tag}.log" 2>&1
}
run_val 0 "$OUT/checkpoints/h1_false_merge_reset_resume1_f0.pt"
run_val 1 "$OUT/checkpoints/h1_false_merge_reset_resume1_f1.pt"
run_val 2 "$OUT/checkpoints/h1_false_merge_reset_resume2_f2.pt"
run_val 3 "$OUT/checkpoints/h1_false_merge_reset_resume3_f3.pt"
echo "H1_TRAIN_VALIDATION_COMPLETE"
