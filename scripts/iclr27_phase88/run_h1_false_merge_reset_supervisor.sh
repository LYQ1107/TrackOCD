#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase88"
CKPT="$OUT/checkpoints"
LOG="$OUT/logs"
mkdir -p "$LOG"

# H1 is a TRAIN-only loss-profile repair from the frozen C0v2 fix2
# checkpoints.  Three workers are used because GPUs 0/4/5/6/7/9 are owned by
# unrelated jobs at launch; fold 3 is run after the first batch completes.
run_one() {
  local fold="$1" gpu="$2"
  local tag="h1_false_merge_reset_f${fold}"
  local done="$OUT/completion/${tag}.done"
  local marker="$OUT/completion/${tag}.launched"
  if [[ -f "$done" ]]; then
    echo "SKIP_DONE $tag"
    RUN_PID=""
    return 0
  fi
  if [[ -f "$marker" ]]; then
    echo "REFUSE_ALREADY_LAUNCHED $tag" >&2
    return 73
  fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=1 \
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device cuda:0 --updates 30000 --tag "$tag" \
      --seed 88002 --event-tag fix2 --memmap-root /data2/usr_for_deadline/trackocd_phase88/shared_features \
      --init-checkpoint "$CKPT/c0v2_fix2_formal_f${fold}.pt" \
      --checkpoint-interval 2000 --loss-profile h1_false_merge_reset \
      >"$LOG/${tag}.log" 2>&1 &
  RUN_PID=$!
}

pids=()
for pair in "0 2" "1 3" "2 8"; do
  read -r fold gpu <<<"$pair"
  run_one "$fold" "$gpu"
  [[ -n "${RUN_PID:-}" ]] && pids+=("$RUN_PID")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" -ne 0 ]]; then
  echo "FIRST_BATCH_FAILED" >&2
  exit "$status"
fi

run_one 3 2
if [[ -n "${RUN_PID:-}" ]]; then
  wait "$RUN_PID"
fi
echo "H1_FORMAL_COMPLETE"
