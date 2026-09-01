#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
TAG="${1:-phase55_formal}"
STAGE="${2:-joint}"
STEPS="${3:-1000}"
BATCH="${BATCH_SIZE:-8}"
INIT_PATTERN="${INIT_CHECKPOINT_PATTERN:-}"
mkdir -p "$ROOT/outputs/iclr27_phase54/logs" "$ROOT/outputs/iclr27_phase54/completion"

pids=()
for fold in 0 1 2 3; do
  gpu=$((fold + 4))
  run="${TAG}_${STAGE}_f${fold}"
  done_file="$ROOT/outputs/iclr27_phase54/completion/${run}.done"
  launched_file="$ROOT/outputs/iclr27_phase54/completion/${run}.launched"
  if [[ -e "$done_file" ]]; then
    continue
  fi
  # A launched marker means the unit is already running or failed and must be
  # inspected/repaired explicitly; never blindly relaunch it.
  if [[ -e "$launched_file" ]]; then
    echo "skip_launched:$run" >&2
    continue
  fi
  extra=()
  if [[ -n "$INIT_PATTERN" ]]; then extra+=(--init-checkpoint "$INIT_PATTERN"); fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase54/train_unified.py" \
    --fold "$fold" --stage "$STAGE" --tag "$TAG" --steps "$STEPS" --batch-size "$BATCH" \
    --checkpoint-every 500 --device cuda:0 --expected-physical-gpu "$gpu" \
    "${extra[@]}" \
    > "$ROOT/outputs/iclr27_phase54/logs/${run}.log" 2>&1 &
  pids+=("$!")
done

# Exactly one blocking wait per spawned worker; no polling loop is used here.
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
printf '%s\n' "${pids[@]}" > "$ROOT/outputs/iclr27_phase54/logs/${TAG}_${STAGE}_worker_pids.txt"
if [[ "$status" -ne 0 ]]; then exit "$status"; fi
printf '{"tag":"%s","stage":"%s","steps":%s,"status":"done"}\n' "$TAG" "$STAGE" "$STEPS" > "$ROOT/outputs/iclr27_phase54/completion/${TAG}_${STAGE}.supervisor.done"
