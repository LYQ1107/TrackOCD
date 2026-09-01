#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
GT="$ROOT/data/external_annotations/ovtr/validation_ours_v1.json"
PREDROOT="$ROOT/outputs/iclr27_phase70/validation/joint_d_repair1"
OUTROOT="$PREDROOT/trackeval"
COMP="$ROOT/outputs/iclr27_phase70/completion"
TAG=joint_d_repair1_trackeval_validation
mkdir -p "$OUTROOT" "$COMP"
{
  echo "phase70_trackeval_preflight=$(date -Iseconds)"
  free -h
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1 /home/user
  echo "gpu_use=none; cpu_trackeval_single_process"
} > "$ROOT/outputs/iclr27_phase70/trackeval_validation_preflight.txt"

status=0
for f in 0 1 2 3; do
  done_marker="$COMP/${TAG}_f${f}.done"
  launched_marker="$COMP/${TAG}_f${f}.launched"
  if [[ -f "$done_marker" ]]; then continue; fi
  if [[ -f "$launched_marker" ]]; then echo "refusing duplicate TrackEval fold${f}" >&2; exit 2; fi
  pred="$PREDROOT/fold${f}_eval/teta_results/tao_track.json"
  out="$OUTROOT/fold${f}"
  [[ -s "$pred" ]] || { echo "missing prediction $pred" >&2; exit 3; }
  printf '{"fold":%d,"tag":"%s","status":"launched","prediction":"%s"}\n' "$f" "$TAG" "$pred" > "$launched_marker.tmp"
  mv -f "$launched_marker.tmp" "$launched_marker"
  if "$PY" "$ROOT/scripts/iclr27_phase69/run_trackeval_fold.py" --fold "$f" --pred-json "$pred" --gt-json "$GT" --out-dir "$out" > "$out.log" 2>&1; then
    printf '{"fold":%d,"tag":"%s","status":"done","out":"%s"}\n' "$f" "$TAG" "$out" > "$done_marker.tmp"
    mv -f "$done_marker.tmp" "$done_marker"
  else
    rc=$?
    status=1
    printf '{"fold":%d,"tag":"%s","status":"failed","exit_code":%d}\n' "$f" "$TAG" "$rc" > "$COMP/${TAG}_f${f}.failed.tmp"
    mv -f "$COMP/${TAG}_f${f}.failed.tmp" "$COMP/${TAG}_f${f}.failed"
    exit "$rc"
  fi
done
exit "$status"
