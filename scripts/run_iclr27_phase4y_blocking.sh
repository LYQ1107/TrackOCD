#!/usr/bin/env bash
# TrackOCD Phase 4Y blocking runner (ADSSI).
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
export PYTHONPATH=.
PY=/home/lwr/anaconda3/envs/locatemot/bin/python

echo "== 00 preflight"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /data1 | tail -1

echo "== 03-04 oracle consistency audit (corrected O1)"
$PY src/iclr27_phase4x/ceilings/oracle_ceilings.py --mode O1 --device cuda:0 \
  --out outputs/iclr27_phase4y/oracle_audit/O1c

echo "== 09-10 anchors + episodes (reuse Phase4W/X artifacts)"
ls outputs/iclr27_phase4x/simple_mixture/known_anchors.npz
ls outputs/iclr27_phase4w/meta_split/capacity.json

echo "== 11-15 Y1 ADSSI training + meta-dev pilot"
$PY src/iclr27_phase4y/training/train_adssi.py --n-episodes 150 --epochs 20 \
  --warmup-epochs 10 --batch-size 4 --lr 3e-4 --fp-entropy-w 0.3 \
  --birth-margin 0.5 --device cuda:0 \
  --out outputs/iclr27_phase4y/y1_dynamic_state/adssi_v2
$PY src/iclr27_phase4y/evaluation/pilot_y.py --split metadev \
  --checkpoint outputs/iclr27_phase4y/y1_dynamic_state/adssi_v2/adssi.pth \
  --n-episodes 200 --device cuda:0 \
  --out outputs/iclr27_phase4y/y1_dynamic_state/pilot_metadev_v2

echo "== 23 Q1 dev (frozen, one shot)"
$PY src/iclr27_phase4y/evaluation/dev_y.py \
  --checkpoint outputs/iclr27_phase4y/y1_dynamic_state/adssi_v2/adssi.pth \
  --device cuda:1 --out outputs/iclr27_phase4y/y1_dynamic_state/dev_v2

echo "== 45 final report"
echo "See docs/iclr27_phase4y/PHASE4Y_COMPLETE_COPYABLE_REPORT.md"
