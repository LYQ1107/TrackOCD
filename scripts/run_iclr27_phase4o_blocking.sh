#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4O blocking orchestrator (unique entry).
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4o"
AUDIT="$ROOT/outputs/iclr27_phase4o/detector_only"
DOC="$ROOT/docs/iclr27_phase4o"
mkdir -p "$RUN_DIR" "$AUDIT" "$DOC"
have() { [ -f "$1" ]; }

echo "== 00-05 preflight + corrected GT =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  | tee "$RUN_DIR/gpu_start.log"
df -h /data1 /data3 > "$RUN_DIR/disk_start.log"
have "$ROOT/outputs/iclr27_phase4n/audit/validation_heldout_tao_corrected.json" \
  || { echo "corrected GT missing"; exit 1; }
have "$ROOT/docs/iclr27_phase4n/PHASE4N_COMPLETE_COPYABLE_REPORT.md" \
  || exit 1

echo "== 06-10 D0 baseline + D1 retraining control =="
have "$AUDIT/D0_current_dev_summary.csv" || {
  $PY -u src/iclr27_phase4o/detector_only_eval.py --name D0_current \
    --mode dev --labeled-csv \
    "$ROOT/outputs/iclr27_phase4n/audit/detection_population_dev.csv" \
    || exit 1; }
have "$AUDIT/D0_current_heldout_summary.csv" || {
  $PY -u src/iclr27_phase4o/detector_only_eval.py --name D0_current \
    --mode heldout --labeled-csv \
    "$ROOT/outputs/iclr27_phase4n/audit/detection_population_heldout_corrected.csv" \
    || exit 1; }
have "$DOC/CURRENT_DETECTOR_RETRAINING_REPORT.md" || exit 1

echo "== 11-25 candidate search/run/eval =="
for f in "$AUDIT/proposals_wedetect_dev.csv" \
  "$AUDIT/proposals_wedetect_heldout.csv" \
  "$AUDIT/proposals_yoloe_dev.csv" \
  "$AUDIT/proposals_yoloe_heldout.csv"; do
  have "$f" || { echo "missing $f"; exit 1; }
done
have "$DOC/DETECTOR_SELECTION_DECISION.md" || exit 1
have "$DOC/NOVEL_RECALL_FP_PARETO.md" || exit 1
have "$DOC/OPEN_SOURCE_DETECTOR_AUDIT.md" || exit 1

echo "== 26-52 re-entry check =="
if grep -q 'NO_DETECTOR_FRONTEND_CLEAR_PROGRESS' \
  "$DOC/DETECTOR_SELECTION_DECISION.md"; then
  echo "no detector pass; end-to-end branch skipped by protocol"
fi
have "$DOC/PHASE4O_COMPLETE_COPYABLE_REPORT.md" || exit 1

echo "== 53-58 tests + report =="
$PY -m pytest tests/iclr27_phase4o/test_phase4o_contracts.py -q \
  -p no:cacheprovider || exit 1
echo "PHASE4O_BLOCKING_DONE"
