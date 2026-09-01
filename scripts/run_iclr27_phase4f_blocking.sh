#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4F blocking orchestrator.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4f"
mkdir -p "$RUN_DIR"
stage_done() { [ -f "$RUN_DIR/.done_$1" ]; }
mark_done() { touch "$RUN_DIR/.done_$1"; }

if ! stage_done audit_md; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/iclr27_phase4f/audit_memory_dynamics.py --stream official --device cuda
  CUDA_VISIBLE_DEVICES=8 $PY -u src/iclr27_phase4f/audit_memory_dynamics.py --stream long --device cuda
  mark_done audit_md
fi

if ! stage_done real_sim; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/iclr27_phase4f/audit_real_similarity.py
  mark_done real_sim
fi

if ! stage_done train_m1; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/train_onpolicy.py --variant M1 \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --mem_scale_norm \
    --update_radius --output_dir mdc_m1
  mark_done train_m1
fi

if ! stage_done train_m2; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/train_onpolicy.py --variant M2 \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --mem_scale_norm \
    --update_radius --real_band_neg_k 2 --output_dir mdc_m2
  mark_done train_m2
fi

if ! stage_done train_m3; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/train_onpolicy.py --variant M3 \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --mem_scale_norm \
    --update_radius --real_band_neg_k 2 --output_dir mdc_m3_q1
  mark_done train_m3
fi

if ! stage_done meta; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/run_meta_dev.py
  mark_done meta
fi

if ! stage_done freeze; then
  $PY -u src/orbit_mdc/freeze_candidates.py
  mark_done freeze
fi

if ! stage_done official; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/run_official.py
  mark_done official
fi

if ! stage_done ablation; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_mdc/run_ablation.py
  mark_done ablation
fi

if ! stage_done tests; then
  $PY -m pytest tests/iclr27_phase4f tests/orbit_mdc -q || true
  mark_done tests
fi
echo "PHASE4F_BLOCKING_DONE"
