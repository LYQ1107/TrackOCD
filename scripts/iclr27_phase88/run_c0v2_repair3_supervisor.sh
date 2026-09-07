#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
LOG="$ROOT/outputs/iclr27_phase88/logs"
CK="$ROOT/outputs/iclr27_phase88/checkpoints"
mkdir -p "$LOG"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export MALLOC_ARENA_MAX=1

run_one() {
  local fold="$1"; local init="$2"
  local tag="c0v2_repair3_f${fold}"
  local marker="$ROOT/outputs/iclr27_phase88/completion/${tag}.launched"
  local done="$ROOT/outputs/iclr27_phase88/completion/${tag}.done"
  if [[ -f "$done" ]]; then echo "skip_done $tag"; return 0; fi
  if [[ -f "$marker" ]]; then echo "refusing_relaunch_launched_without_done $tag" >&2; return 2; fi
  [[ -f "$init" ]] || { echo "missing_init_checkpoint fold=$fold init=$init" >&2; return 3; }
  # 25% of 125 GiB is the hard floor; this is a single bounded worker.
  local avail_kib
  avail_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( avail_kib < 32000000 )); then
    echo "memory_floor_abort fold=$fold available_kib=$avail_kib" >&2
    return 4
  fi
  echo "resume tag=$tag fold=$fold gpu=5 init=$init available_kib=$avail_kib"
  "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
    --fold "$fold" --device cuda:5 --updates 20000 --tag "$tag" \
    --checkpoint-interval 2000 --resume-checkpoint "$init" \
    >"$LOG/${tag}.log" 2>&1
}

run_one 0 "$CK/c0v2_repair1_f0_step006000.pt"
run_one 1 "$CK/c0v2_formal_f1_step006000.pt"
run_one 2 "$CK/c0v2_formal_f2_step004000.pt"
run_one 3 "$CK/c0v2_formal_f3_step006000.pt"
echo "repair3_supervisor_complete"
