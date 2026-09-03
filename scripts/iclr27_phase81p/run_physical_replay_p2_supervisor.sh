#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-newlabel}"
ROUTE="${2:-p2_newlabel}"
MOTION="${3:-0}"
MAX_MISS="${4:-8}"
APPEARANCE="${5:-0}"
mkdir -p "$ROOT/outputs/iclr27_phase81p/completion" "$ROOT/outputs/iclr27_phase81p/metrics"
declare -a PIDS=() GPUS=(4 5 6 7)
for fold in 0 1 2 3; do
  gpu="${GPUS[$fold]}"
  done_m="$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.done"
  launched="$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.launched"
  [[ -f "$done_m" ]] && continue
  [[ -f "$launched" ]] && { echo "skip already launched physical fold=$fold route=$ROUTE"; continue; }
  python - "$fold" "$gpu" "$TAG" >"$launched" <<'PY'
import json,sys,os,datetime
print(json.dumps({'phase':'Phase81P+','route':'p2_newlabel','fold':int(sys.argv[1]),'gpu':int(sys.argv[2]),'tag':sys.argv[3],'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
PY
  log="/data2/usr_for_deadline/trackocd_phase81p/physical_${ROUTE}_${TAG}_f${fold}.log"
  eval_args=(--checkpoint "$ROOT/outputs/iclr27_phase81p/checkpoints/${ROUTE}/fold${fold}/best.pt" --device cuda:0 --tag "${ROUTE}_${TAG}_f${fold}" --max-miss "$MAX_MISS")
  [[ "$MOTION" == "1" ]] && eval_args+=(--motion)
  [[ "$APPEARANCE" == "1" ]] && eval_args+=(--appearance)
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase81p/evaluate_physical_replay.py" "${eval_args[@]}" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "launched physical fold=$fold gpu=$gpu pid=${PIDS[-1]} route=$ROUTE"
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  for fold in 0 1 2 3; do tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${ROUTE}_${TAG}_f${fold}.done.tmp"; printf 'complete\n' >"$tmp"; mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}_f${fold}.done"; done
  tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${ROUTE}_${TAG}.done.tmp"; printf 'supervisor_complete route=%s tag=%s\n' "$ROUTE" "$TAG" >"$tmp"; mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}.done"
else
  printf 'supervisor_failed route=%s tag=%s\n' "$ROUTE" "$TAG" >"$ROOT/outputs/iclr27_phase81p/completion/physical_${ROUTE}_${TAG}.failed"
fi
exit "$status"
