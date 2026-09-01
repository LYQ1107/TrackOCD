#!/usr/bin/env bash
# TrackOCD Phase 4W blocking runner (genuine-OOV cold/warm state machine).
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
export PYTHONPATH=.
PY=/home/lwr/anaconda3/envs/locatemot/bin/python

echo "== 00 preflight"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /data1 | tail -1

echo "== 02 verify fixed harness"
rg -n "MAX_OCC = 24" src/iclr27_phase4t/episodes.py

echo "== 06-08 capacity / split / active universe"
$PY src/iclr27_phase4w/audit_category_capacity.py
$PY src/iclr27_phase4w/active_universe/build_prototypes.py --device cuda:0

echo "== 09 genuine OOV leakage tests"
$PY src/iclr27_phase4w/audits/leakage_test.py

echo "== 10-11 build cold/warm samples + heads"
$PY src/iclr27_phase4w/episodes/build_episodes.py \
  --out outputs/iclr27_phase4w/cold_start/samples_train400_v3 \
  --split train --n-episodes 400 --device cuda:0 \
  --known-set-sizes 2,3,4,6,8,10,12
$PY src/iclr27_phase4w/episodes/build_episodes.py \
  --out outputs/iclr27_phase4w/meta_split/samples_metadev200_v2 \
  --split metadev --n-episodes 200 --device cuda:1
$PY src/iclr27_phase4w/cold_start/train.py --head cold \
  --samples outputs/iclr27_phase4w/cold_start/samples_train400_v3/samples.npz \
  --meta-dev-samples outputs/iclr27_phase4w/meta_split/samples_metadev200_v2/samples.npz \
  --out outputs/iclr27_phase4w/cold_start/head_cold_v3 --epochs 100 --device cuda:0
$PY src/iclr27_phase4w/cold_start/train.py --head warm \
  --samples outputs/iclr27_phase4w/cold_start/samples_train400_v3/samples.npz \
  --meta-dev-samples outputs/iclr27_phase4w/meta_split/samples_metadev200_v2/samples.npz \
  --out outputs/iclr27_phase4w/warm_memory/head_warm_v3 --epochs 100 --device cuda:1

echo "== 12-16 pilots"
$PY src/iclr27_phase4w/evaluation/pilot.py --split metadev --n-episodes 200 \
  --out outputs/iclr27_phase4w/full_rollout/pilot_metadev_v3 --device cuda:0
$PY src/iclr27_phase4w/evaluation/pilot.py --split train --n-episodes 200 \
  --out outputs/iclr27_phase4w/full_rollout/pilot_train_v3 --device cuda:1

echo "== 11 simple episodic-OOD + clustering baseline"
$PY src/iclr27_phase4w/baselines/simple_ood_clustering.py --n-episodes 200 \
  --out outputs/iclr27_phase4w/baselines/simple_ood_clustering --device cuda:2

echo "== 22 Q1 dev eval (frozen candidate, one shot)"
$PY src/iclr27_phase4w/evaluation/dev_eval.py \
  --out outputs/iclr27_phase4w/full_rollout/dev_v3 --device cuda:0

echo "== 41 final report"
echo "See docs/iclr27_phase4w/PHASE4W_COMPLETE_COPYABLE_REPORT.md"
