#!/usr/bin/env bash
# Train and evaluate Phase 7A ablations (dev only; heldout is locked).
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python
cd "$ROOT"

run_abl() {
  local name=$1 gpu=$2; shift 2
  local dir=$ROOT/outputs/iclr27_phase7a/training/$name
  if [[ ! -f "$dir/best.pth" ]]; then
    PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 "$PY" \
      src/iclr27_phase7a/training/train_reliability_head.py \
      --epochs 30 --device cuda:0 --out "outputs/iclr27_phase7a/training/$name" \
      --w-novel 20 --w-unlabeled 0.05 --known-tau 0.65 \
      --visible-known-per-chunk 17 "$@" \
      > "outputs/iclr27_phase7a/training/$name.log" 2>&1
  fi
}

run_abl abl_no_rel 7 --no-rel &
P1=$!
run_abl abl_no_maturity 9 --no-maturity &
P2=$!
wait $P1 $P2

run_abl abl_no_cross_track 7 --no-cross-track
run_abl abl_sem_only 9 --sem-only

bash scripts/run_iclr27_phase7a_eval.sh 7 \
  outputs/iclr27_phase7a/training/abl_no_rel/best.pth abl_no_rel racc --no-rel
bash scripts/run_iclr27_phase7a_eval.sh 9 \
  outputs/iclr27_phase7a/training/abl_no_maturity/best.pth abl_no_maturity racc --no-maturity
bash scripts/run_iclr27_phase7a_eval.sh 7 \
  outputs/iclr27_phase7a/training/abl_no_cross_track/best.pth abl_no_cross_track racc
bash scripts/run_iclr27_phase7a_eval.sh 9 \
  outputs/iclr27_phase7a/training/abl_sem_only/best.pth abl_sem_only racc --sem-only

echo "PHASE7A_ABLATIONS_DONE"
