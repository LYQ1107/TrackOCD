#!/usr/bin/env bash
# V0 non-parametric baseline replay + strict eval on the Q1 stream.
set -euo pipefail

GPU=${1:?gpu}
NAME=${2:?name}
shift 2

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PHYS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv
FEATS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/feats.npz
OUT=$ROOT/outputs/iclr27_phase6c/eval/$NAME
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
PY_BASE=/home/lwr/anaconda3/bin/python
mkdir -p "$OUT"

PYTHONPATH="$ROOT" "$PY_BASE" "$ROOT/src/iclr27_phase6c/evaluation/v0_baseline.py" \
  --proposals "$PHYS" \
  --feats "$FEATS" \
  --out-csv "outputs/iclr27_phase6c/eval/$NAME/proposals_sem.csv" \
  "$@" \
  > "$OUT/replay.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase5a/evaluation/strict_causal_eval.py" \
  --proposals "outputs/iclr27_phase6c/eval/$NAME/proposals_sem.csv" \
  --feats "$FEATS" \
  --proto-dir outputs/iclr27_phase5a/pilot/episodes \
  --embed h --mode jointcsv --filter aligned \
  --device "cuda:$GPU" \
  --out "outputs/iclr27_phase6c/eval/$NAME/strict" \
  > "$OUT/strict.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase6a/evaluation/physical_eval.py" \
  --csv "outputs/iclr27_phase6c/eval/$NAME/proposals_sem.csv" \
  --out "outputs/iclr27_phase6c/eval/$NAME/physical.json" \
  > "$OUT/physical.log" 2>&1

echo "V0_EVAL_DONE $NAME"
