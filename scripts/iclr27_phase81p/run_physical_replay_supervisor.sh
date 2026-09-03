#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-formal}"
mkdir -p "$ROOT/outputs/iclr27_phase81p/completion" "$ROOT/outputs/iclr27_phase81p/metrics"
declare -a PIDS=() GPUS=(4 5 6 7)
for fold in 0 1 2 3; do
  gpu="${GPUS[$fold]}"
  done_m="$ROOT/outputs/iclr27_phase81p/completion/physical_${TAG}_f${fold}.done"
  launched="$ROOT/outputs/iclr27_phase81p/completion/physical_${TAG}_f${fold}.launched"
  [[ -f "$done_m" ]] && continue
  [[ -f "$launched" ]] && { echo "skip already launched physical replay fold=$fold"; continue; }
  python - "$fold" "$gpu" "$TAG" >"$launched" <<'PY'
import json,sys
print(json.dumps({"phase":"Phase81P+","fold":int(sys.argv[1]),"gpu":int(sys.argv[2]),"tag":sys.argv[3]}))
PY
  log="/data2/usr_for_deadline/trackocd_phase81p/physical_${TAG}_f${fold}.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase81p/evaluate_physical_replay.py" \
    --checkpoint "$ROOT/outputs/iclr27_phase81p/checkpoints/fold${fold}/best.pt" --device cuda:0 --tag "${TAG}_f${fold}" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "launched fold=$fold gpu=$gpu pid=${PIDS[-1]}"
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  for fold in 0 1 2 3; do
    tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${TAG}_f${fold}.done.tmp"
    printf 'complete\n' >"$tmp"; mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${TAG}_f${fold}.done"
  done
  tmp="$ROOT/outputs/iclr27_phase81p/completion/.physical_${TAG}.done.tmp"
  printf 'supervisor_complete %s\n' "$TAG" >"$tmp"; mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/completion/physical_${TAG}.done"
else
  printf 'supervisor_failed %s\n' "$TAG" >"$ROOT/outputs/iclr27_phase81p/completion/physical_${TAG}.failed"
fi
exit "$status"
