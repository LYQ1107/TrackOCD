#!/usr/bin/env bash
# TrackOCD Phase 4V blocking runner (frozen dual-space first version).
# Reproduces the exact experiment chain used in docs/iclr27_phase4v.
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
export PYTHONPATH=.
PY=/home/lwr/anaconda3/envs/locatemot/bin/python

echo "== 00 preflight"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /data1 | tail -1

echo "== 01 read history"
head -5 AGENTS.md

echo "== 02 verify fixed harness (full 24-occurrence)"
rg -n "MAX_OCC = 24" src/iclr27_phase4t/episodes.py

echo "== 06-07 frozen evidence cache (optional; reproduces samples)"
$PY src/iclr27_phase4v/router_data.py \
  --out outputs/iclr27_phase4v/router_pilot/samples_400 \
  --n-episodes 400 --seed 20260815 --device cuda:0

$PY src/iclr27_phase4v/router_data.py \
  --out outputs/iclr27_phase4v/router_pilot/samples_400_perstep \
  --n-episodes 400 --seed 20260815 --device cuda:0 --all-steps

$PY src/iclr27_phase4v/router_data.py \
  --out outputs/iclr27_phase4v/router_pilot/samples_400_masked \
  --n-episodes 400 --seed 20260815 --device cuda:0 --all-steps

echo "== 08 known/novel separability audit"
$PY src/iclr27_phase4v/evidence_audit.py \
  --samples outputs/iclr27_phase4v/router_pilot/samples_400/samples.npz \
  --out outputs/iclr27_phase4v/evidence_audit/separability.json

echo "== 10-11 dual evidence routers"
$PY src/iclr27_phase4v/train_router.py \
  --samples outputs/iclr27_phase4v/router_pilot/samples_400_masked/samples.npz \
  --out outputs/iclr27_phase4v/router_pilot/router_mlp_masked \
  --arch mlp --epochs 60 --device cuda:0
$PY src/iclr27_phase4v/train_router.py \
  --samples outputs/iclr27_phase4v/router_pilot/samples_400_masked/samples.npz \
  --out outputs/iclr27_phase4v/router_pilot/router_logistic_masked \
  --arch logistic --epochs 60 --device cuda:1

echo "== 12-13 episodic pilot + dev"
$PY src/iclr27_phase4v/pilot.py \
  --router outputs/iclr27_phase4v/router_pilot/router_mlp_masked/router.pth \
  --out outputs/iclr27_phase4v/dualspace_pilot/pilot_mlp_masked \
  --n-episodes 300 --device cuda:0
$PY src/iclr27_phase4v/dev_eval.py \
  --router outputs/iclr27_phase4v/router_pilot/router_mlp_masked/router.pth \
  --out outputs/iclr27_phase4v/dualspace_pilot/dev_mlp_masked \
  --device cuda:1

echo "== 09 simple OOD baselines (diagnostic frontier)"
$PY src/iclr27_phase4v/ood_baseline_dev.py \
  --out outputs/iclr27_phase4v/baselines/ood_cls_energy_t0 \
  --evidence cls_energy --min-age 0 \
  '--thresholds=4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2' --device cuda:0

echo "== 35-37 report"
echo "See docs/iclr27_phase4v/PHASE4V_COMPLETE_COPYABLE_REPORT.md"
