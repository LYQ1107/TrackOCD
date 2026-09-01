#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4G blocking orchestrator.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4g"
mkdir -p "$RUN_DIR"
stage_done() { [ -f "$RUN_DIR/.done_$1" ]; }
mark_done() { touch "$RUN_DIR/.done_$1"; }

if ! stage_done preflight; then
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv > "$RUN_DIR/gpu_start.log"
  free -h > "$RUN_DIR/mem_start.log"
  mkdir -p configs/iclr27_phase4g configs/orbit_msrouting \
    docs/iclr27_phase4g docs/orbit_msrouting outputs/iclr27_phase4g \
    outputs/orbit_msrouting runs/iclr27_phase4g runs/orbit_msrouting \
    src/iclr27_phase4g src/orbit_msrouting tests/iclr27_phase4g \
    tests/orbit_msrouting third_party/research_refs_phase4g
  mark_done preflight
fi

if ! stage_done threshold_pareto; then
  CUDA_VISIBLE_DEVICES=5 $PY -u src/orbit_msrouting/scan_threshold.py \
    --checkpoint runs/orbit_mdc/mdc_m2/model.pth \
    --out outputs/iclr27_phase4g/audit/static_threshold_pareto.csv \
    --device cuda --gate_thrs 0.4,0.45,0.5,0.55
  mark_done threshold_pareto
fi

if ! stage_done train_g1; then
  CUDA_VISIBLE_DEVICES=5 $PY -u src/orbit_msrouting/train_msrouting.py \
    --variant G1 --gate_mode G1 --state_feats \
    log_mem,low_support_ratio,mean_support,recent_birth_rate,high_disp_ratio \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --update_radius \
    --real_band_neg_k 2 --output_dir msrouting_g1
  mark_done train_g1
fi

if ! stage_done train_g2; then
  CUDA_VISIBLE_DEVICES=6 $PY -u src/orbit_msrouting/train_msrouting.py \
    --variant G2 --gate_mode G2 --state_feats \
    log_mem,low_support_ratio,mean_support,recent_birth_rate,high_disp_ratio \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --update_radius \
    --real_band_neg_k 2 --output_dir msrouting_g2
  mark_done train_g2
fi

if ! stage_done train_g1_memonly; then
  CUDA_VISIBLE_DEVICES=7 $PY -u src/orbit_msrouting/train_msrouting.py \
    --variant G1_MEMONLY --gate_mode G1 --state_feats log_mem \
    --epochs 24 --episodes_per_epoch 6 --balanced --margin --update_radius \
    --real_band_neg_k 2 --output_dir msrouting_g1_memonly
  mark_done train_g1_memonly
fi

if ! stage_done meta_dev; then
  CUDA_VISIBLE_DEVICES=9 $PY -u src/orbit_msrouting/run_meta_dev.py \
    --checkpoints runs/orbit_mdc/mdc_m2/model.pth \
    runs/orbit_msrouting/msrouting_g1/model.pth \
    runs/orbit_msrouting/msrouting_g2/model.pth \
    runs/orbit_msrouting/msrouting_g1_memonly/model.pth \
    --device cuda --compat_threshold 0.45 --compat_margin 0.05
  mark_done meta_dev
fi

echo "PHASE4G_BLOCKING_PART1_DONE"
