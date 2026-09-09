#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase88"
CKPT="$OUT/checkpoints"
LOG="$OUT/logs"
MEMMAP="/data2/usr_for_deadline/trackocd_phase88/shared_features"
mkdir -p "$LOG"

run_one() {
  local fold="$1"
  local source="$2"
  local tag="h2_known_suppression_cpu_formal_f${fold}"
  if [[ -f "$OUT/completion/${tag}.done" ]]; then
    echo "SKIP_DONE $tag"
    return 0
  fi
  if [[ -f "$OUT/completion/${tag}.launched" ]]; then
    echo "REFUSE_ALREADY_LAUNCHED $tag" >&2
    return 73
  fi
  TRACKOCD_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=1 \
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device cpu --updates 30000 --tag "$tag" \
      --seed 88002 --event-tag fix2 --memmap-root "$MEMMAP" \
      --resume-checkpoint "$source" --checkpoint-interval 2000 \
      --loss-profile h2_known_suppression \
      >"$LOG/${tag}.log" 2>&1 &
  RUN_PID=$!
}

pids=()
run_one 0 "$CKPT/h2_known_suppression_resume1_f0_step022000.pt"; pids+=("$RUN_PID")
run_one 1 "$CKPT/h2_known_suppression_f1_step020000.pt"; pids+=("$RUN_PID")
run_one 2 "$CKPT/h2_known_suppression_f2_step018000.pt"; pids+=("$RUN_PID")
run_one 3 "$CKPT/c0v2_fix2_formal_f3.pt"; pids+=("$RUN_PID")
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "H2_CPU_FORMAL_FAILED" >&2; exit "$status"; }
echo "H2_CPU_FORMAL_COMPLETE"
