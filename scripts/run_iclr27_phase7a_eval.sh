#!/usr/bin/env bash
# Phase 7A evaluation of one RACC head checkpoint on Q1 DEV and the locked
# heldout split. Only semantic columns are recomputed on the frozen
# physical stream; strict causal metrics reuse the Phase 5A definitions.
set -euo pipefail

GPU=${1:?gpu}
CKPT=${2:?racc head checkpoint}
NAME=${3:?run name}
MODE=${4:-racc}
EXTRA=${5:-}

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python
DEV_CSV=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv
DEV_FEATS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/feats.npz
HO_CSV=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/proposals_heldout.csv
HO_FEATS=$ROOT/outputs/iclr27_phase7a/assets/q1_heldout/feats.npz
DEV_VIDEOS='[88,90,122,291,334,888,931,1159,1232,1276,1572,1865,2254,2347,2564,2675,2690,2759,2802,2888]'
HO_VIDEOS='[37,49,207,642,669,771,820,882,1047,1313,1361,1364,1589,1701,1797,1842,1877,2169,2329,2510,2527,2780,2826,2915]'
JOINT_STATS=$ROOT/outputs/iclr27_phase6b/q1/final_dsct/joint_stats.json
OUT=$ROOT/outputs/iclr27_phase7a/eval/$NAME
mkdir -p "$OUT"

HEAD_ARGS=()
if [[ "$MODE" == "racc" ]]; then
  HEAD_ARGS=(--mode racc --head-ckpt "$CKPT")
else
  HEAD_ARGS=(--mode ema --tau 0.65)
fi

for SPLIT in dev heldout; do
  if [[ "$SPLIT" == dev ]]; then
    CSV=$DEV_CSV; FEATS=$DEV_FEATS; VIDS=$DEV_VIDEOS
  else
    CSV=$HO_CSV; FEATS=$HO_FEATS; VIDS=$HO_VIDEOS
  fi
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
    "$ROOT/src/iclr27_phase7a/evaluation/replay_online.py" \
    --proposals "$CSV" --feats "$FEATS" \
    --out-csv "$OUT/${SPLIT}_sem.csv" \
    "${HEAD_ARGS[@]}" --device "cuda:0" $EXTRA \
    > "$OUT/${SPLIT}_replay.log" 2>&1
  PYTHONPATH="$ROOT" "$PY" \
    "$ROOT/src/iclr27_phase7a/evaluation/strict_eval_any.py" \
    --proposals "$OUT/${SPLIT}_sem.csv" --feats "$FEATS" \
    --video-ids "$VIDS" --device cpu \
    --out "$OUT/${SPLIT}_strict" \
    > "$OUT/${SPLIT}_strict.log" 2>&1
  PYTHONPATH="$ROOT" "$PY" \
    "$ROOT/src/iclr27_phase7a/evaluation/physical_stats_any.py" \
    --csv "$OUT/${SPLIT}_sem.csv" --out "$OUT/${SPLIT}_physical.json" \
    > "$OUT/${SPLIT}_physical.log" 2>&1
  PYTHONPATH="$ROOT" "$PY" \
    "$ROOT/src/iclr27_phase6a/tests/causal_contract_tests.py" \
    --csv "$OUT/${SPLIT}_sem.csv" --out "$OUT/${SPLIT}_contract.json" \
    > "$OUT/${SPLIT}_contract.log" 2>&1
done

if [[ "$MODE" == "racc" ]]; then
  PYTHONPATH="$ROOT" "$PY" \
    "$ROOT/src/iclr27_phase6a/evaluation/objectness_audit.py" \
    --joint-stats "$JOINT_STATS" --out "$OUT/objectness_audit.json" \
    > "$OUT/objectness.log" 2>&1 || true
fi

echo "PHASE7A_EVAL_DONE $NAME"
