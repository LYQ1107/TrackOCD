#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/ovtr/bin/python"; TAG="${1:-formal_replay}"
mkdir -p "$ROOT/outputs/iclr27_phase81p/completion" "$ROOT/outputs/iclr27_phase81p/metrics"
declare -a PIDS=(); declare -a GPUS=(4 5 6 7)
for fold in 0 1 2 3; do
  gpu="${GPUS[$fold]}"; done_m="$ROOT/outputs/iclr27_phase81p/completion/replay_${TAG}_f${fold}.done"; launched="$ROOT/outputs/iclr27_phase81p/completion/replay_${TAG}_f${fold}.launched"
  [[ -f "$done_m" ]] && continue
  [[ -f "$launched" ]] && { echo "skip already launched replay fold=$fold"; continue; }
  printf '%s\n' "$(python -c 'import json,sys; print(json.dumps({"phase":"Phase81P+","fold":int(sys.argv[1]),"gpu":int(sys.argv[2]),"tag":sys.argv[3]}))' "$fold" "$gpu" "$TAG")" > "$launched"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase81p/replay_association.py" --checkpoint "$ROOT/outputs/iclr27_phase81p/checkpoints/fold${fold}/best.pt" --device cuda:0 --tag "${TAG}_f${fold}" >"/data2/usr_for_deadline/trackocd_phase81p/replay_${TAG}_f${fold}.log" 2>&1 & PIDS+=("$!")
done
status=0; for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done
if [[ "$status" -eq 0 ]]; then
  for fold in 0 1 2 3; do printf 'complete\n' > "$ROOT/outputs/iclr27_phase81p/completion/replay_${TAG}_f${fold}.done"; done
  printf 'supervisor_complete %s\n' "$TAG" > "$ROOT/outputs/iclr27_phase81p/completion/replay_${TAG}.done"
else
  printf 'supervisor_failed %s\n' "$TAG" > "$ROOT/outputs/iclr27_phase81p/completion/replay_${TAG}.failed"
fi
exit "$status"
