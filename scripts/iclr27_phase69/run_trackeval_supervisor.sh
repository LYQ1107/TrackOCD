#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON_BIN:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
OUT="$ROOT/outputs/iclr27_phase69"
GT="$ROOT/data/external_annotations/ovtr/validation_ours_v1.json"
PRED_ROOT="$OUT/metrics"
MARK="$OUT/completion"
mkdir -p "$MARK" "$OUT/metrics"
{
  date -Is
  printf 'cwd=%s\n' "$ROOT"
  free -h
  nvidia-smi || true
  printf 'process_count='; ps -e --no-headers | wc -l
  df -h "$ROOT"
} > "$OUT/trackeval_preflight.txt"
for f in 0 1 2 3; do
  done_marker="$MARK/fold${f}_trackeval.done"
  launched="$MARK/fold${f}_trackeval.launched"
  failed="$MARK/fold${f}_trackeval.failed"
  if test -f "$done_marker"; then continue; fi
  if test -f "$launched"; then
    echo "fold $f already launched without done; refusing duplicate" >&2
    exit 2
  fi
  printf '%s\n' "pid=$$ fold=$f" > "$launched.tmp"
  mv "$launched.tmp" "$launched"
  pred="$PRED_ROOT/fold${f}_eval/teta_results/tao_track.json"
  out="$OUT/trackeval/fold${f}"
  if "$PY" "$ROOT/scripts/iclr27_phase69/run_trackeval_fold.py" --fold "$f" --pred-json "$pred" --gt-json "$GT" --out-dir "$out" > "$OUT/trackeval_fold${f}.log" 2>&1; then
    test -s "$out/run_note.json"
    printf 'complete\n' > "$done_marker.tmp"
    mv "$done_marker.tmp" "$done_marker"
  else
    status=$?
    printf 'exit=%s\n' "$status" > "$failed.tmp"
    mv "$failed.tmp" "$failed"
    exit "$status"
  fi
done
echo "phase69 trackeval complete"
