#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4N blocking orchestrator (unique entry).
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4n"
AUDIT="$ROOT/outputs/iclr27_phase4n/audit"
DOC="$ROOT/docs/iclr27_phase4n"
mkdir -p "$RUN_DIR" "$AUDIT" "$DOC"
have() { [ -f "$1" ]; }
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
      $2==0 && $3==0 {print $1; exit}'
}

echo "== 00 preflight =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  | tee "$RUN_DIR/gpu_start.log"
free -h > "$RUN_DIR/mem_start.log"
df -h /data1 /data3 >> "$RUN_DIR/mem_start.log"

echo "== 01-03 inputs =="
have "$ROOT/AGENTS.md" || exit 1
have "$ROOT/docs/iclr27_phase4m/PHASE4M_COMPLETE_COPYABLE_REPORT.md" || {
  echo "phase4m report missing"; exit 1; }

echo "== 04-05 corrected held-out GT =="
have "$AUDIT/validation_heldout_tao_corrected.json" || {
  $PY -u src/iclr27_phase4n/correct_heldout_gt.py || exit 1; }

echo "== 08-13 frontend audits =="
for f in detection_population_dev.csv \
  detection_population_heldout_corrected.csv \
  detector_score_distributions.csv detector_threshold_curve.csv \
  persistent_fp_features.csv validity_predictability.csv; do
  have "$AUDIT/$f" || { echo "missing $f"; exit 1; }
done
$PY -u src/iclr27_phase4n/write_audit_docs.py || exit 1
have "$DOC/FRONTEND_ROOT_DECISION.md" || exit 1

echo "== 14-20 gate audits =="
for f in gate_scores_dev.csv gate_scores_heldout.csv \
  gate_shift_by_age.csv gate_shift_by_video.csv \
  detector_gate_interaction.csv gate_shift_summary.csv; do
  have "$AUDIT/$f" || { echo "missing $f"; exit 1; }
done
have "$DOC/GATE_ROOT_DECISION.md" || exit 1

echo "== 21-24 open-source audit =="
have "$AUDIT/../open_source/repository_inventory.csv" || exit 1
have "$DOC/OPEN_SOURCE_REPOSITORY_AUDIT.md" || exit 1

echo "== 25-28 conditional methods (calibration skipped; validity tested) =="
if grep -q 'VALIDITY_AWARE_ROUTING_SUPPORTED' \
  "$DOC/DEVELOPMENT_RESULTS.md"; then
  echo "validity branch supported (further dev work required)"
fi

echo "== 37-44 component decision + freeze =="
have "$ROOT/outputs/iclr27_phase4n/dev/component_comparison.csv" || exit 1
grep -q 'VALIDITY_AWARE_ROUTING_NOT_SUPPORTED' \
  "$DOC/DEVELOPMENT_RESULTS.md" && \
  echo "N2 failed pass gate; no candidate frozen"

echo "== 45-53 corrected held-out re-evaluation =="
for tag in j1b m1 m3; do
  have "$AUDIT/identity_decisions_ho_${tag}_corrected.csv" || {
    echo "missing identity decisions $tag"; exit 1; }
done
have "$DOC/HELDOUT_RESULTS.md" || exit 1
have "$DOC/GENERALIZATION_DECISION.md" || exit 1

echo "== 56-60 novelty + tests + report =="
have "$DOC/PHASE4N_METHOD_NOVELTY_AUDIT.md" || exit 1
have "$DOC/PHASE4N_COMPLETE_COPYABLE_REPORT.md" || exit 1
$PY -m pytest tests/iclr27_phase4n/test_phase4n_contracts.py -q \
  -p no:cacheprovider || exit 1
echo "PHASE4N_BLOCKING_DONE"
