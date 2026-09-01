#!/usr/bin/env bash
# One blocking proposal-crop feature -> frozen-B semantic audit.
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
GPU=${PHASE14C_GPU:-5}
LOG=$ROOT/outputs/iclr27_phase14c/logs
mkdir -p "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/src/iclr27_phase14c/representation/extract_proposal_dinov2.py" \
  --device cuda:0 > "$LOG/extract_proposal_dinov2.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/src/iclr27_phase14c/representation/survival.py" \
  --device cuda:0 > "$LOG/representation_survival.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase14c/representation/train_only_normalize.py" \
  --in outputs/iclr27_phase14c/features/proposal_dinov2.npz \
  --out outputs/iclr27_phase14c/features/proposal_dinov2_trainnorm.npz \
  --meta outputs/iclr27_phase14c/features/train_only_normalization.json \
  > "$LOG/train_only_normalization.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/src/iclr27_phase8a/evaluation/replay_amortized.py" \
  --proposals outputs/iclr27_phase14c/proposals/proposals_mixed.csv \
  --feats outputs/iclr27_phase14c/features/proposal_dinov2.npz \
  --adapter-ckpt outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth \
  --out-csv outputs/iclr27_phase14c/eval/frozen_b_exact.csv --device cuda:0 \
  > "$LOG/frozen_b_exact_replay.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase14c/evaluation/strict_mixed.py" \
  --proposals outputs/iclr27_phase14c/eval/frozen_b_exact.csv \
  --out outputs/iclr27_phase14c/eval/frozen_b_exact_summary.json \
  > "$LOG/frozen_b_exact_strict.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" \
  "$ROOT/src/iclr27_phase8a/evaluation/replay_amortized.py" \
  --proposals outputs/iclr27_phase14c/proposals/proposals_mixed.csv \
  --feats outputs/iclr27_phase14c/features/proposal_dinov2_trainnorm.npz \
  --adapter-ckpt outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth \
  --out-csv outputs/iclr27_phase14c/eval/frozen_b_trainnorm.csv --device cuda:0 \
  > "$LOG/frozen_b_trainnorm_replay.log" 2>&1

PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase14c/evaluation/strict_mixed.py" \
  --proposals outputs/iclr27_phase14c/eval/frozen_b_trainnorm.csv \
  --out outputs/iclr27_phase14c/eval/frozen_b_trainnorm_summary.json \
  > "$LOG/frozen_b_trainnorm_strict.log" 2>&1

echo "PHASE14C_SEMANTIC_AUDIT_DONE"
