#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4H blocking orchestrator (unique entry).
#
# Idempotent: each stage runs only if its artifact is missing. Run with:
#   bash scripts/run_iclr27_phase4h_blocking.sh 2>&1 |
#     tee runs/iclr27_phase4h/phase4h_blocking.log
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4h"
CHP_RUN="$ROOT/runs/orbit_chp"
mkdir -p "$RUN_DIR" "$CHP_RUN"

pick_gpu() {
  # strictly free GPUs only (0 MiB, 0% util); no fallback to busy cards.
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
      $2==0 && $3==0 {print $1; exit}'
}

have() { [ -f "$1" ]; }

echo "== Phase 4H preflight =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  | tee "$RUN_DIR/gpu_start.log"
free -h > "$RUN_DIR/mem_start.log"
df -h /data1 >> "$RUN_DIR/mem_start.log"
mkdir -p configs/iclr27_phase4h configs/orbit_chp docs/iclr27_phase4h \
  docs/orbit_chp outputs/iclr27_phase4h outputs/orbit_chp \
  runs/iclr27_phase4h runs/orbit_chp src/iclr27_phase4h src/orbit_chp \
  tests/iclr27_phase4h tests/orbit_chp third_party/research_refs_phase4h

echo "== 01-04 inputs and gate audit =="
if ! have outputs/iclr27_phase4h/audit/input_hashes.json; then
  echo "input manifest missing; materialize from Phase 4F/4G first"
  exit 1
fi
[ -f docs/iclr27_phase4h/GATE_COMPUTATION_GRAPH_AUDIT.md ] || \
  { echo "gate audit doc missing"; exit 1; }

echo "== 05-12 root-cause audit =="
for f in raw_feature_hardness_by_bucket adapted_feature_hardness_by_bucket \
         novel_class_hardness permutation_results counterfactual_memory_replay \
         root_cause_probe; do
  [ -f "outputs/iclr27_phase4h/audit/$f.csv" ] || \
    { echo "missing audit artifact $f.csv"; exit 1; }
done
[ -f docs/iclr27_phase4h/ROOT_CAUSE_DECISION.md ] || \
  { echo "missing root cause decision"; exit 1; }

echo "== 13-15 open-source review =="
for f in repository_inventory mechanism_matrix; do
  [ -f "outputs/iclr27_phase4h/open_source/$f.csv" ] || \
    { echo "missing $f.csv"; exit 1; }
done

echo "== 16-23 CHP training =="
for spec in "H1 random chp_h1" "H2 hard chp_h2" "H3 mixed chp_h3"; do
  set -- $spec
  V=$1; M=$2; O=$3
  if ! have "$CHP_RUN/$O/model.pth"; then
    G=$(pick_gpu)
    [ -n "$G" ] || { echo "no free GPU for $V; stop"; exit 1; }
    CUDA_VISIBLE_DEVICES=$G $PY -u src/orbit_chp/train_chp.py \
      --variant "$V" --episode_mode "$M" \
      --epochs 24 --episodes_per_epoch 6 --real_band_neg_k 2 \
      --output_dir "$O"
  else
    echo "skip $V (checkpoint exists)"
  fi
done

echo "== 24 held-out + long-stream evaluation =="
if ! have outputs/orbit_chp/meta_dev/model_comparison.csv; then
  G=$(pick_gpu)
  [ -n "$G" ] || { echo "no free GPU for eval; stop"; exit 1; }
  CUDA_VISIBLE_DEVICES=$G $PY -u src/orbit_chp/eval_proxy.py \
    --checkpoints runs/orbit_mdc/mdc_m2/model.pth \
    runs/orbit_chp/chp_h1/model.pth \
    runs/orbit_chp/chp_h2/model.pth \
    runs/orbit_chp/chp_h3/model.pth \
    --names chp_h0 random_leaveout hard_leaveout mixed_curriculum \
    --compat_threshold 0.45 --compat_margin 0.05
fi

echo "== 26-34 decision / freeze / official / representation =="
if ! have outputs/iclr27_phase4h/audit/representation_separability_summary.json; then
  CUDA_VISIBLE_DEVICES= $PY -u src/iclr27_phase4h/representation_limit.py
fi
# No candidate was frozen in Phase 4H (see
# docs/orbit_chp/OFFICIAL_CANDIDATE_FREEZE.md), so no official seed1027 run
# is authorized.

echo "== 36-37 tests =="
$PY -m pytest tests/iclr27_phase4h tests/orbit_chp -q \
  --disable-warnings > "$RUN_DIR/pytest.log" 2>&1 || \
  { tail -20 "$RUN_DIR/pytest.log"; exit 1; }
tail -3 "$RUN_DIR/pytest.log"

echo "== 38-39 status and reports =="
[ -f docs/iclr27_phase4h/PHASE4H_COMPLETE_COPYABLE_REPORT.md ] || \
  { echo "final report missing"; exit 1; }
echo "PHASE4H_BLOCKING_DONE"
