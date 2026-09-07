#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
LOG="$ROOT/outputs/iclr27_phase88/logs"
CK="$ROOT/outputs/iclr27_phase88/checkpoints"
mkdir -p "$LOG"
run_pair() {
  local f0="$1"; local f1="$2"; local g0="$3"; local g1="$4"
  local -a pids=()
  for spec in "$f0:$g0" "$f1:$g1"; do
    local fold="${spec%%:*}"; local gpu="${spec##*:}"
    local tag="c0v2_repair2_f${fold}"
    local marker="$ROOT/outputs/iclr27_phase88/completion/${tag}.launched"
    local done="$ROOT/outputs/iclr27_phase88/completion/${tag}.done"
    if [[ -f "$done" ]]; then echo "skip_done $tag"; continue; fi
    if [[ -f "$marker" ]]; then echo "refusing_relaunch_launched_without_done $tag" >&2; exit 2; fi
    local init
    init=$(find "$CK" -maxdepth 1 -type f \( -name "c0v2_repair1_f${fold}_step*.pt" -o -name "c0v2_formal_f${fold}_step*.pt" \) -printf '%p\n' | sort | tail -1)
    [[ -n "$init" ]] || { echo "missing_init_checkpoint fold=$fold" >&2; exit 3; }
    echo "resume tag=$tag fold=$fold gpu=$gpu init=$init"
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device "cuda:${gpu}" --updates 20000 --tag "$tag" \
      --checkpoint-interval 2000 --resume-checkpoint "$init" \
      >"$LOG/${tag}.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  return "$status"
}
run_pair 0 1 5 6
run_pair 2 3 7 8
echo "repair2_supervisor_complete"
