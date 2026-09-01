#!/usr/bin/env bash
# Phase 4M single blocking entry point.
# Usage:
#   bash scripts/run_iclr27_phase4m_blocking.sh \
#       2>&1 | tee runs/iclr27_phase4m/phase4m_blocking.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD
PY=/home/lwr/anaconda3/envs/locatemot/bin/python

step() { echo "[phase4m] $*"; }

step 00_preflight
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /data1 /data3 | tail -2

step 01_read_agents
cat AGENTS.md >/dev/null

step 03_input_manifest
ls outputs/iclr27_phase4l/audit/prov_dev_j1b/prototype_event_log_j1b.jsonl >/dev/null
ls outputs/iclr27_phase4m/audit/det_z_cache >/dev/null

step 06_build_identity_decision_dataset
for tag in j1b b1 b2; do
  $PY -u src/iclr27_phase4m/build_identity_decision_dataset.py --tag "$tag" --gpu -1
done

step 10-13_audits
$PY -u src/iclr27_phase4m/run_phase4m_audits.py --tags j1b,b1,b2
$PY src/iclr27_phase4m/generate_phase4m_docs.py

step 14-17_open_source
ls outputs/iclr27_phase4m/open_source/repository_inventory.csv >/dev/null
ls outputs/iclr27_phase4m/open_source/mechanism_matrix.csv >/dev/null

step 18-21_dev_replay_and_eval
for tag in m0 m1 m2 m3; do
  $PY -u src/iclr27_phase4m/run_dev_candidates.py --tag "$tag" --gpu 0
  $PY -u src/iclr27_phase4m/eval_dev_candidates.py --tag "$tag"
done

step 30_freeze
echo "frozen candidates: m1"

step 31-33_heldout
for tag in m0 m1; do
  $PY -u src/iclr27_phase4m/run_heldout_replay.py --tag "$tag" --gpu 0
  $PY -u src/iclr27_phase4m/run_heldout_replay.py --tag "$tag" --eval
done

step 42_tests
$PY -m pytest tests/iclr27_phase4m -q

step 43-44_report
$PY src/iclr27_phase4m/generate_phase4m_docs.py
echo "PHASE4M_BLOCKING_DONE"
