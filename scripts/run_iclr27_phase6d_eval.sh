#!/usr/bin/env bash
# Phase 6D evaluation on the frozen Q1 physical stream (Phase 6B DSCT final).
set -euo pipefail

GPU=${1:?gpu}
CKPT=${2:?gmna checkpoint}
NAME=${3:?name}

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PHYS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv
FEATS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/feats.npz
JOINT_STATS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/joint_stats.json
OUT=$ROOT/outputs/iclr27_phase6d/eval/$NAME
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
mkdir -p "$OUT"

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase6c/evaluation/online_semantic_eval.py" \
  --proposals "$PHYS" \
  --feats "$FEATS" \
  --ckpt "$CKPT" \
  --out-csv "outputs/iclr27_phase6d/eval/$NAME/proposals_sem.csv" \
  --calibrate \
  --device "cuda:$GPU" \
  > "$OUT/replay.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase5a/evaluation/strict_causal_eval.py" \
  --proposals "outputs/iclr27_phase6d/eval/$NAME/proposals_sem.csv" \
  --feats "$FEATS" \
  --proto-dir outputs/iclr27_phase5a/pilot/episodes \
  --embed h --mode jointcsv --filter aligned \
  --device "cuda:$GPU" \
  --out "outputs/iclr27_phase6d/eval/$NAME/strict" \
  > "$OUT/strict.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase6a/evaluation/physical_eval.py" \
  --csv "outputs/iclr27_phase6d/eval/$NAME/proposals_sem.csv" \
  --out "outputs/iclr27_phase6d/eval/$NAME/physical.json" \
  > "$OUT/physical.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase6a/tests/causal_contract_tests.py" \
  --csv "outputs/iclr27_phase6d/eval/$NAME/proposals_sem.csv" \
  --out "outputs/iclr27_phase6d/eval/$NAME/causal_contract.json" \
  > "$OUT/contract.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase6a/evaluation/objectness_audit.py" \
  --joint-stats "$JOINT_STATS" \
  --out "outputs/iclr27_phase6d/eval/$NAME/objectness_audit.json" \
  > "$OUT/objectness.log" 2>&1

echo "PHASE6D_EVAL_DONE $NAME"
