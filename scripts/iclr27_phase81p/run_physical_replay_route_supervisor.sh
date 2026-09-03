#!/usr/bin/env bash
set -euo pipefail

# Bounded physical replay supervisor for a route-specific checkpoint root.
# Each fold gets one explicitly mapped GPU and writes completion markers only
# after its atomic metric output is present.  The evaluator joins GT only
# after causal inference, so this script never changes the runtime contract.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
ROUTE="${1:?route name required}"
TAG="${2:-physical}"
CKPT_ROOT="${3:?checkpoint root required}"
MOTION="${4:-0}"
MAX_MISS="${5:-8}"
APPEARANCE="${6:-0}"
RESOLUTION_AWARE="${7:-0}"

mkdir -p "$ROOT/outputs/iclr27_phase81p/completion" "$ROOT/outputs/iclr27_phase81p/metrics"
declare -a PIDS=() GPUS=(4 5 6 7)
for fold in 0 1 2 3; do
  gpu="${GPUS[$fold]}"
  done_m="$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.done"
  launched="$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.launched"
  [[ -f "$done_m" ]] && continue
  [[ -f "$launched" ]] && { echo "skip already launched physical replay fold=$fold route=$ROUTE"; continue; }
  python - "$fold" "$gpu" "$TAG" "$ROUTE" >"$launched" <<'PY'
import datetime, json, os, sys
print(json.dumps({
    "phase": "Phase81P+",
    "route": sys.argv[4],
    "fold": int(sys.argv[1]),
    "gpu": int(sys.argv[2]),
    "tag": sys.argv[3],
    "pid": os.getpid(),
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}))
PY
  log="/data2/usr_for_deadline/trackocd_phase81p/physical_${ROUTE}_${TAG}_f${fold}.log"
  eval_args=(
    --checkpoint "$CKPT_ROOT/fold${fold}/best.pt"
    --device cuda:0
    --tag "${ROUTE}_${TAG}_f${fold}"
    --max-miss "$MAX_MISS"
  )
  [[ "$MOTION" == "1" ]] && eval_args+=(--motion)
  [[ "$APPEARANCE" == "1" ]] && eval_args+=(--appearance)
  [[ "$RESOLUTION_AWARE" == "1" ]] && eval_args+=(--resolution-aware)
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" \
    "$ROOT/scripts/iclr27_phase81p/evaluate_physical_replay.py" \
    "${eval_args[@]}" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "launched physical replay fold=$fold gpu=$gpu pid=${PIDS[-1]} route=$ROUTE"
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -eq 0 ]]; then
  for fold in 0 1 2 3; do
    metric="$ROOT/outputs/iclr27_phase81p/metrics/physical_${ROUTE}_${TAG}_f${fold}.json"
    [[ -s "$metric" ]] || { echo "missing metric fold=$fold" >&2; status=1; }
  done
fi
if [[ "$status" -eq 0 ]]; then
  for fold in 0 1 2 3; do
    tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${ROUTE}_${TAG}_f${fold}.done.tmp"
    printf 'complete\n' >"$tmp"
    mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.done"
  done
  tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${ROUTE}_${TAG}.done.tmp"
  printf 'supervisor_complete route=%s tag=%s\n' "$ROUTE" "$TAG" >"$tmp"
  mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}.done"
else
  printf 'supervisor_failed route=%s tag=%s\n' "$ROUTE" "$TAG" >"$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}.failed"
fi
exit "$status"
