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
  local gpu="$2"
  local tag="h2_known_suppression_f${fold}"
  if [[ -f "$OUT/completion/${tag}.done" ]]; then echo "SKIP_DONE $tag"; RUN_PID=""; return 0; fi
  if [[ -f "$OUT/completion/${tag}.launched" ]]; then echo "REFUSE_ALREADY_LAUNCHED $tag" >&2; return 73; fi
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MALLOC_ARENA_MAX=1 \
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device cuda:0 --updates 30000 --tag "$tag" \
      --seed 88002 --event-tag fix2 --memmap-root "$MEMMAP" \
      --init-checkpoint "$CKPT/c0v2_fix2_formal_f${fold}.pt" \
      --checkpoint-interval 2000 --loss-profile h2_known_suppression \
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
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "FIRST_BATCH_FAILED" >&2; exit "$status"; }
run_one 3 2
if [[ -n "${RUN_PID:-}" ]]; then wait "$RUN_PID"; fi
echo "H2_FORMAL_COMPLETE"
