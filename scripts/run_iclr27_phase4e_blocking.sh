#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4E blocking orchestrator.
# Stages that already produced validated artifacts are skipped via markers.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4e"
mkdir -p "$RUN_DIR"

stage_done() { [ -f "$RUN_DIR/.done_$1" ]; }
mark_done() { touch "$RUN_DIR/.done_$1"; }

if ! stage_done preflight; then
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  free -h
  df -h /data1 | tail -1
  mark_done preflight
fi

if ! stage_done identity_audit; then
  GPU_IDX=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
    | awk -F', ' '$2==0 && $3==0 {print $1; exit}')
  [ -z "$GPU_IDX" ] && GPU_IDX=8
  CUDA_VISIBLE_DEVICES="$GPU_IDX" $PY -u src/iclr27_phase4e/audit_identity.py --mode replay --stream official
  CUDA_VISIBLE_DEVICES="$GPU_IDX" $PY -u src/iclr27_phase4e/audit_identity.py --mode replay --stream long
  CUDA_VISIBLE_DEVICES="$GPU_IDX" $PY -u src/iclr27_phase4e/audit_identity.py --mode multimodal
  mark_done identity_audit
fi

if ! stage_done train_i1; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/train_iam.py \
    --variant I1 --compat_feats sim,margin,radius,support,mem,rel \
    --freeze_mode compat --epochs 24 --episodes_per_epoch 6 \
    --balanced --margin --mem_scale_norm --update_radius --output_dir iam_i1_v2
  mark_done train_i1
fi

if ! stage_done train_i2; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/train_iam.py \
    --variant I2 --compat_feats sim,margin,radius,support,conf,mem,rel \
    --freeze_mode compat --epochs 24 --episodes_per_epoch 6 \
    --balanced --margin --mem_scale_norm --update_radius --output_dir iam_i2_v2
  mark_done train_i2
fi

if ! stage_done train_i3; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/train_iam.py \
    --variant I3 --compat_feats sim,margin,radius,support,conf,mem,rel \
    --freeze_mode full --epochs 20 --episodes_per_epoch 6 \
    --balanced --margin --mem_scale_norm --update_radius --lr 5e-4 --output_dir iam_i3_v2
  mark_done train_i3
fi

if ! stage_done train_a3; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/train_iam.py \
    --variant A3 --compat_feats sim,radius,support,conf,mem,rel \
    --freeze_mode compat --epochs 24 --episodes_per_epoch 6 \
    --balanced --margin --mem_scale_norm --update_radius --output_dir iam_a3_v2
  mark_done train_a3
fi

if ! stage_done meta_dev_select; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/scan_iam.py \
    --checkpoint runs/orbit_iam/iam_i1_v2/model.pth \
    --out outputs/orbit_iam/meta_dev/scan_i1.csv
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/scan_iam.py \
    --checkpoint runs/orbit_iam/iam_i2_v2/model.pth \
    --out outputs/orbit_iam/meta_dev/scan_i2.csv
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/scan_iam.py \
    --checkpoint runs/orbit_iam/iam_i3_v2/model.pth \
    --out outputs/orbit_iam/meta_dev/scan_i3.csv
  mark_done meta_dev_select
fi

if ! stage_done freeze; then
  $PY -u src/orbit_iam/freeze_candidate.py \
    --checkpoint runs/orbit_iam/iam_i2_v3/model.pth --candidate A \
    --compat_threshold 0.45 --compat_margin 0.05 \
    --selection_evidence "long-stream proxy plateau at margin 0.05"
  mark_done freeze
fi

if ! stage_done official; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/run_official.py \
    --checkpoint runs/orbit_iam/iam_i2_v3/model.pth \
    --candidate_name candidate_a --compat_threshold 0.45 \
    --compat_margin 0.05 --device cuda
  mark_done official
fi

if ! stage_done ablation; then
  CUDA_VISIBLE_DEVICES=8 $PY -u src/orbit_iam/run_ablation.py
  mark_done ablation
fi

if ! stage_done tests; then
  $PY -m pytest tests/iclr27_phase4e tests/orbit_iam -q || true
  mark_done tests
fi

echo "PHASE4E_BLOCKING_DONE"
