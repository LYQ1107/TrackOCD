#!/usr/bin/env bash
# Phase 4U blocking runner (documentation / reproducible sequence).
# Idempotent: each step writes a .done marker and is skipped if present.
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
cd "$ROOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"

run() {
  local name="$1"; shift
  if [ -f "outputs/iclr27_phase4u/.done_${name}" ]; then
    echo "[skip] $name"
    return
  fi
  echo "[run] $name"
  "$@"
  touch "outputs/iclr27_phase4u/.done_${name}"
}

mkdir -p outputs/iclr27_phase4u

# 02 data capacity + raw crop audit (already executed)
run data_audit env PYTHONPATH=. "$PY" src/iclr27_phase4u/data_audit.py \
  --out outputs/iclr27_phase4u/data_audit.json

# 07-08 build cross-track pairs + retrieval benchmark (frozen baselines)
run frozen_baselines env PYTHONPATH=. "$PY" src/iclr27_phase4u/bench.py \
  --out outputs/iclr27_phase4u/representation/frozen_baselines.json \
  --sources real,episodic,dev --prefix-lengths 1,2,3,4,6,8,12,16

# 12 representation pilot (R3 cross-track SupCon, mixed universe, GRU)
run r3_pilot env PYTHONPATH=. "$PY" src/iclr27_phase4u/pretrain/train_rep.py \
  --out outputs/iclr27_phase4u/representation/r3_mixed_gru \
  --source mixed --arch gru --steps 3000 --n-classes 16 --k-per-class 3 \
  --device cuda:0

# 13 geometry gate
run r3_eval env PYTHONPATH=. "$PY" src/iclr27_phase4u/evaluation/eval_rep.py \
  --checkpoint outputs/iclr27_phase4u/representation/r3_mixed_gru/checkpoint.pth \
  --out outputs/iclr27_phase4u/representation/r3_mixed_gru/geometry_all.json \
  --sources real,episodic,dev --device cuda:0

# 14-19 downstream T3 retrain (frozen representation, heads only)
run d1_full env PYTHONPATH=. "$PY" src/iclr27_phase4u/downstream/train.py \
  --out outputs/iclr27_phase4u/downstream/d1_full_fixed --data real \
  --use-hierarchy --epochs 40 --episodes-per-epoch 256 --batch-size 4 \
  --device cuda:0 \
  --tsr-checkpoint outputs/iclr27_phase4u/representation/r3_mixed_gru/checkpoint.pth

run d1_pilot env PYTHONPATH=. "$PY" src/iclr27_phase4u/downstream/pilot.py \
  --checkpoint outputs/iclr27_phase4u/downstream/d1_full_fixed/checkpoint.pth \
  --data real --use-hierarchy --n-episodes 300 \
  --out outputs/iclr27_phase4u/downstream/d1_full_fixed_pilot --device cuda:0 \
  --tsr-checkpoint outputs/iclr27_phase4u/representation/r3_mixed_gru/checkpoint.pth

run d1_dev env PYTHONPATH=. "$PY" src/iclr27_phase4u/downstream/dev_eval.py \
  --checkpoint outputs/iclr27_phase4u/downstream/d1_full_fixed/checkpoint.pth \
  --use-hierarchy --out outputs/iclr27_phase4u/downstream/d1_full_fixed_dev \
  --device cuda:0 \
  --tsr-checkpoint outputs/iclr27_phase4u/representation/r3_mixed_gru/checkpoint.pth

# 26-29 ablations
for ab in none same_phys; do
  run "abl_${ab}" env PYTHONPATH=. "$PY" src/iclr27_phase4u/pretrain/train_rep.py \
    --out "outputs/iclr27_phase4u/representation/abl_${ab}" \
    --source mixed --arch gru --supervision "${ab}" \
    --steps 3000 --n-classes 16 --k-per-class 3 --device cuda:0
done

run real_only env PYTHONPATH=. "$PY" src/iclr27_phase4u/pretrain/train_rep.py \
  --out outputs/iclr27_phase4u/representation/r3_real_only_gru \
  --source real --arch gru --steps 3000 --n-classes 16 --k-per-class 3 \
  --device cuda:0

echo "Phase 4U blocking runner done."
