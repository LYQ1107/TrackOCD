#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-newlabel}"
EPOCHS="${2:-20}"
DATA_ROOT="${3:-/data2/usr_for_deadline/trackocd_phase81p/data/p2_newlabel}"
ROUTE="p2_newlabel"
mkdir -p "$ROOT/outputs/iclr27_phase81p/supervisor" "$ROOT/outputs/iclr27_phase81p/completion"
declare -a PIDS=() GPUS=(4 5 6 7)
for fold in 0 1 2 3; do
  gpu="${GPUS[$fold]}"
  done_m="$ROOT/outputs/iclr27_phase81p/completion/association_${ROUTE}_${TAG}_f${fold}.done"
  launched="$ROOT/outputs/iclr27_phase81p/completion/association_${ROUTE}_${TAG}_f${fold}.launched"
  [[ -f "$done_m" ]] && continue
  [[ -f "$launched" ]] && { echo "skip already launched fold=$fold route=$ROUTE"; continue; }
  python - "$fold" "$gpu" "$TAG" "$ROUTE" >"$launched" <<'PY'
import json,sys,os,datetime
print(json.dumps({'phase':'Phase81P+','route':sys.argv[4],'fold':int(sys.argv[1]),'gpu':int(sys.argv[2]),'tag':sys.argv[3],'pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
PY
  log="/data2/usr_for_deadline/trackocd_phase81p/${ROUTE}_${TAG}_f${fold}.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase81p/train_association.py" \
    --fold "$fold" --device cuda:0 --tag "$TAG" --route "$ROUTE" --data-root "$DATA_ROOT" --epochs "$EPOCHS" >"$log" 2>&1 &
  PIDS+=("$!")
  echo "launched fold=$fold gpu=$gpu pid=${PIDS[-1]} route=$ROUTE"
done
status=0
for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  tmp="$ROOT/outputs/iclr27_phase81p/supervisor/.${ROUTE}_${TAG}.done.tmp"; printf 'supervisor_complete route=%s tag=%s\n' "$ROUTE" "$TAG" >"$tmp"; mv -f "$tmp" "$ROOT/outputs/iclr27_phase81p/supervisor/${ROUTE}_${TAG}.done"
else
  printf 'supervisor_failed route=%s tag=%s\n' "$ROUTE" "$TAG" >"$ROOT/outputs/iclr27_phase81p/supervisor/${ROUTE}_${TAG}.failed"
fi
exit "$status"
