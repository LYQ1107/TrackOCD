#!/usr/bin/env bash
# TrackOCD ICLR 2027 - Phase 4Z blocking runner (reproducibility entry point).
# Each step is a blocking shell command; long jobs run once and wait.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
PYA=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python

echo "[00] preflight"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
free -h
df -h /data1 | tail -1

echo "[02] verify corrected harness"
PYTHONPATH=. $PY src/iclr27_phase4s/frontend.py >/tmp/p4z_frontend.log 2>&1 || true

echo "[05] reproduce O1c ceiling"
PYTHONPATH=. $PY src/iclr27_phase4x/ceilings/oracle_ceilings.py --mode O1 \
  --device cuda:0 --out outputs/iclr27_phase4z/final/O1c >/tmp/p4z_o1c.log 2>&1

echo "[06] full per-step routing dump"
PYTHONPATH=. $PY src/iclr27_phase4z/routing_audit/dump_routing_full.py \
  --device cuda:0 --out outputs/iclr27_phase4z/routing_audit/dev_dump_full \
  >/tmp/p4z_dump.log 2>&1

echo "[07-09] mechanism audit"
PYTHONPATH=. $PYA src/iclr27_phase4z/routing_audit/analyze_routing.py \
  --dump outputs/iclr27_phase4z/routing_audit/dev_dump_full \
  --out outputs/iclr27_phase4z/routing_audit/audit
PYTHONPATH=. $PYA src/iclr27_phase4z/routing_audit/audit_detail.py

echo "[15-17] build episodes (full, no frozen-L1 evidence)"
PYTHONPATH=. $PY src/iclr27_phase4z/episodes/build_phase4z_episodes.py \
  --split train --n-episodes 400 --seed 20260815 \
  --out outputs/iclr27_phase4z/episodes/full_train46_nol1 \
  --device cuda:0 --max-len 12 --known-set-sizes 4,6
PYTHONPATH=. $PY src/iclr27_phase4z/episodes/build_phase4z_episodes.py \
  --split metadev --n-episodes 200 --seed 20260816 \
  --out outputs/iclr27_phase4z/episodes/full_metadev46_nol1 \
  --device cuda:0 --max-len 12 --known-set-sizes 4,6

echo "[18/21] full training (candidates; max 4 GPUs)"
PYTHONPATH=. $PY src/iclr27_phase4z/training/train_routing.py --mode gru \
  --train-episodes outputs/iclr27_phase4z/episodes/full_train46_nol1/episodes.npz \
  --meta-dev-episodes outputs/iclr27_phase4z/episodes/full_metadev46_nol1/episodes.npz \
  --out outputs/iclr27_phase4z/full/gru_nol1 --epochs 40 --seed 20260815 \
  --device cuda:0 --normalize 0 --label-scheme flat --hidden 128
PYTHONPATH=. $PY src/iclr27_phase4z/training/train_routing.py --mode static \
  --train-episodes outputs/iclr27_phase4z/episodes/full_train46_nol1/episodes.npz \
  --meta-dev-episodes outputs/iclr27_phase4z/episodes/full_metadev46_nol1/episodes.npz \
  --out outputs/iclr27_phase4z/full/static_nol1 --epochs 40 --seed 20260815 \
  --device cuda:6 --normalize 0 --label-scheme flat
PYTHONPATH=. $PY src/iclr27_phase4z/training/train_routing.py --mode meanpool \
  --train-episodes outputs/iclr27_phase4z/episodes/full_train46_nol1/episodes.npz \
  --meta-dev-episodes outputs/iclr27_phase4z/episodes/full_metadev46_nol1/episodes.npz \
  --out outputs/iclr27_phase4z/full/meanpool_nol1 --epochs 40 --seed 20260815 \
  --device cuda:7 --normalize 0 --label-scheme flat
PYTHONPATH=. $PY src/iclr27_phase4z/training/train_routing.py --mode aggregated \
  --train-episodes outputs/iclr27_phase4z/episodes/full_train46_nol1/episodes.npz \
  --meta-dev-episodes outputs/iclr27_phase4z/episodes/full_metadev46_nol1/episodes.npz \
  --out outputs/iclr27_phase4z/full/aggregated_nol1 --epochs 40 --seed 20260815 \
  --device cuda:8 --normalize 0 --label-scheme flat

echo "[23-30] Q1 dev end-to-end (frozen candidate + frozen downstream)"
PYTHONPATH=. $PY src/iclr27_phase4z/evaluation/end_to_end.py \
  --router outputs/iclr27_phase4z/full/gru_nol1 --tau 0.45 --device cuda:0 \
  --out outputs/iclr27_phase4z/final/dev_gru_nol1 --frontend q1

echo "[31-35] ablations (meta-dev)"
PYTHONPATH=. $PY src/iclr27_phase4z/ablations/ablations.py \
  --router outputs/iclr27_phase4z/full/gru_nol1 \
  --meta-dev-episodes outputs/iclr27_phase4z/episodes/full_metadev46_nol1/episodes.npz \
  --tau 0.45 --device cuda:0 --out outputs/iclr27_phase4z/ablations/gru_nol1

echo "[38-41] cross-frontend + diagnostics after candidate freeze"
PYTHONPATH=. $PY src/iclr27_phase4z/evaluation/end_to_end.py \
  --router outputs/iclr27_phase4z/full/gru_nol1 --tau 0.45 --device cuda:0 \
  --out outputs/iclr27_phase4z/cross_frontend/dev_q2_gru_nol1 --frontend q2

echo "[43-44] report and link"
echo "Phase 4Z complete. See docs/iclr27_phase4z/PHASE4Z_COMPLETE_COPYABLE_REPORT.md"
