#!/usr/bin/env bash
set -euo pipefail
# One bounded supervisor: exactly one worker per fixed GPU 4/5/6/7.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-formal}"
EPOCHS="${2:-20}"
MAX_STEPS="${3:-}"
mkdir -p "$ROOT/outputs/iclr27_phase81p/supervisor"
declare -a PIDS=()
declare -a FOLDS=(0 1 2 3)
declare -a GPUS=(4 5 6 7)
for i in "${!FOLDS[@]}"; do
  fold="${FOLDS[$i]}"; gpu="${GPUS[$i]}"
  done_marker="$ROOT/outputs/iclr27_phase81p/completion/association_${TAG}_f${fold}.done"
  launched_marker="$ROOT/outputs/iclr27_phase81p/completion/association_${TAG}_f${fold}.launched"
  if [[ -f "$done_marker" ]]; then continue; fi
  if [[ -f "$launched_marker" ]]; then echo "skip already launched fold=$fold tag=$TAG"; continue; fi
  log="/data2/usr_for_deadline/trackocd_phase81p/${TAG}_f${fold}.log"
  args=(--fold "$fold" --device cuda:0 --tag "$TAG" --epochs "$EPOCHS")
  [[ -n "$MAX_STEPS" ]] && args+=(--max-steps "$MAX_STEPS")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase81p/train_association.py" "${args[@]}" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "launched fold=$fold gpu=$gpu pid=${PIDS[-1]} tag=$TAG"
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  printf 'supervisor_complete tag=%s\n' "$TAG" > "$ROOT/outputs/iclr27_phase81p/supervisor/${TAG}.done"
else
  printf 'supervisor_failed tag=%s\n' "$TAG" > "$ROOT/outputs/iclr27_phase81p/supervisor/${TAG}.failed"
fi
exit "$status"
