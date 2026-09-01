#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
GPU=${PHASE14C_GPU:-5}
LOG=$ROOT/outputs/iclr27_phase14c/logs

for spec in exact trainnorm; do
  if [[ "$spec" == exact ]]; then feats=outputs/iclr27_phase14c/features/proposal_dinov2.npz; out=outputs/iclr27_phase14c/eval/frozen_b_exact.csv; slog=frozen_b_exact; else feats=outputs/iclr27_phase14c/features/proposal_dinov2_trainnorm.npz; out=outputs/iclr27_phase14c/eval/frozen_b_trainnorm.csv; slog=frozen_b_trainnorm; fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase8a/evaluation/replay_amortized.py" \
    --proposals outputs/iclr27_phase14c/proposals/proposals_mixed.csv --feats "$feats" \
    --adapter-ckpt outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth --out-csv "$out" --device cuda:0 \
    > "$LOG/${slog}_replay.log" 2>&1
  PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase14c/evaluation/strict_mixed.py" \
    --proposals "$out" --out "outputs/iclr27_phase14c/eval/${slog}_summary.json" \
    > "$LOG/${slog}_strict.log" 2>&1
done
echo "PHASE14C_REPLAY_ONLY_DONE"
