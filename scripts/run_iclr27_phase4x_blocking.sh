#!/usr/bin/env bash
# TrackOCD Phase 4X blocking runner (semantic formulation reset).
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
export PYTHONPATH=.
PY=/home/lwr/anaconda3/envs/locatemot/bin/python

echo "== 00 preflight"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /data1 | tail -1

echo "== 02 verify corrected harness"
rg -n "MAX_OCC = 24" src/iclr27_phase4t/episodes.py

echo "== 03-05 oracle ceilings"
$PY src/iclr27_phase4x/ceilings/oracle_ceilings.py --mode O1 --device cuda:0 \
  --out outputs/iclr27_phase4x/ceilings/O1
$PY src/iclr27_phase4x/ceilings/oracle_ceilings.py --mode O2 --device cuda:0 \
  --out outputs/iclr27_phase4x/ceilings/O2
$PY src/iclr27_phase4x/ceilings/oracle_ceilings.py --mode O3 --device cuda:1 \
  --out outputs/iclr27_phase4x/ceilings/O3

echo "== 10-11 geometry audit + anchors"
$PY src/iclr27_phase4x/components/build_anchors.py --device cuda:0

echo "== 12-14 X3 calibration + pilot"
$PY src/iclr27_phase4x/evaluation/calibrate_x3.py --n-episodes 40 \
  --device cuda:0 --out outputs/iclr27_phase4x/simple_mixture/calibration2.json
$PY src/iclr27_phase4x/evaluation/pilot_x3.py --split metadev --n-episodes 200 \
  --device cuda:0 --out outputs/iclr27_phase4x/simple_mixture/pilot_metadev_k8 \
  --kappa 8 --log-prior-new -1.5 --log-prior-noise -3.0 --noise-alpha 2.0 \
  --commit-threshold 0.5 --margin-ratio 1.5 --min-age 2

echo "== 15-17 X4 learned compatibility"
$PY src/iclr27_phase4x/likelihood/train_compatibility.py --n-episodes 400 \
  --epochs 30 --device cuda:0 --out outputs/iclr27_phase4x/learned_predictive/compat_v1
$PY src/iclr27_phase4x/evaluation/pilot_x3.py --split metadev --n-episodes 200 \
  --device cuda:0 --out outputs/iclr27_phase4x/learned_predictive/pilot_metadev_x4 \
  --compat outputs/iclr27_phase4x/learned_predictive/compat_v1/compat.pth

echo "== 20-24 Q1 dev (frozen, one shot each)"
$PY src/iclr27_phase4x/evaluation/dev_x3.py \
  --out outputs/iclr27_phase4x/simple_mixture/dev_k8 --device cuda:0 \
  --kappa 8 --log-prior-new -1.5 --log-prior-noise -3.0 --noise-alpha 2.0 \
  --commit-threshold 0.5 --margin-ratio 1.5 --min-age 2
$PY src/iclr27_phase4x/evaluation/dev_x3.py \
  --out outputs/iclr27_phase4x/learned_predictive/dev_x4 --device cuda:1 \
  --compat outputs/iclr27_phase4x/learned_predictive/compat_v1/compat.pth

echo "== 44 online clustering control"
$PY src/iclr27_phase4x/ablations/online_clustering_control.py --n-episodes 60 \
  --device cuda:0 --out outputs/iclr27_phase4x/ablations/online_clustering

echo "== 46 final report"
echo "See docs/iclr27_phase4x/PHASE4X_COMPLETE_COPYABLE_REPORT.md"
