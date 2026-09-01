#!/usr/bin/env bash
# Phase 4S TrackOCD semantic core: blocking pipeline.
# Usage: bash scripts/run_iclr27_phase4s_blocking.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export CUDA_VISIBLE_DEVICES=0

OUT=outputs/iclr27_phase4s

echo "[00] preflight"
free -h | head -2
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -4

echo "[01] train semantic core (blocking wait)"
$PY src/iclr27_phase4s/train.py \
  --out "$OUT/full_model" \
  --epochs 20 --episodes-per-epoch 256 --batch-size 4 --num-workers 2 \
  --device cuda:0

echo "[02] episodic pilot gate"
$PY src/iclr27_phase4s/pilot.py \
  --checkpoint "$OUT/full_model/checkpoint.pth" \
  --n-episodes 300 --out "$OUT/episodic_pilot" --device cuda:0

echo "[03] extract Q1 dev semantic features (blocking wait)"
$PY src/iclr27_phase4s/features_q1.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --out "$OUT/q1_features" --device cuda:0 --batch 64

echo "[04] TrackOCD dev evaluation B0/B1/B2/B3"
for mode in b0 b1 b2 b3; do
  $PY src/iclr27_phase4s/dev_eval.py \
    --checkpoint "$OUT/full_model/checkpoint.pth" \
    --feats "$OUT/q1_features/feats.npz" \
    --mode "$mode" --out "$OUT/dev_eval" --device cuda:0
done

echo "[05] cross-frontend Q2-alpha0.1 check"
$PY src/iclr27_phase4s/features_q1.py \
  --proposals outputs/iclr27_phase4r/q2_alpha/a010/proposals_dev.csv \
  --out "$OUT/q2_features" --device cuda:0 --batch 64
$PY src/iclr27_phase4s/dev_eval.py \
  --checkpoint "$OUT/full_model/checkpoint.pth" \
  --proposals outputs/iclr27_phase4r/q2_alpha/a010/proposals_dev.csv \
  --feats "$OUT/q2_features/feats.npz" \
  --mode b3 --out "$OUT/dev_eval_q2" --device cuda:0

echo "[06] final report"
$PY src/iclr27_phase4s/generate_report.py
echo "PHASE4S_PIPELINE_DONE"
