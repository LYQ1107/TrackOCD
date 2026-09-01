#!/usr/bin/env bash
# ICLR27 Phase 2.5 blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/iclr27_phase2_5/markers"
mkdir -p "$MK" "$ROOT/runs/iclr27_phase2_5" "$ROOT/outputs/iclr27_phase2_5/audit" \
  "$ROOT/outputs/iclr27_phase2_5/tables" "$ROOT/outputs/iclr27_phase2_5/analysis" \
  "$ROOT/outputs/iclr27_phase2_5/end_to_end" "$ROOT/outputs/iclr27_phase2_5/tests"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/outputs/simowt/val_predictions.json" ]]; finish 00_preflight; }
s01() { if ! begin 01_read_agents; then return 0; fi
  [[ -f "$ROOT/AGENTS.md" ]]; finish 01_read_agents; }
s02() { if ! begin 02_read_phase2; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2/PHASE2_FINAL_REPORT.md" ]]; finish 02_read_phase2; }
s03() { if ! begin 03_hash_inputs; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/audit/input_hashes.json" ]]; finish 03_hash_inputs; }
s04() { if ! begin 04_prediction_semantics; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/audit/prediction_semantics.csv" ]]; finish 04_prediction_semantics; }
s05() { if ! begin 05_duplicate_analysis; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/audit/duplicate_prediction_analysis.csv" ]]; finish 05_duplicate_analysis; }
s06() { if ! begin 06_gt_count_reconstruction; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/audit/gt_count_reconstruction.csv" ]]; finish 06_gt_count_reconstruction; }
s07() { if ! begin 07_metric_scale_audit; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/tables/simowt_metrics_paper_scale.csv" ]]; finish 07_metric_scale_audit; }
s08() { if ! begin 08_deta_error_decomposition; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/analysis/deta_error_decomposition.csv" ]]; finish 08_deta_error_decomposition; }
s09() { if ! begin 09_frame_track_coverage_gap; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/analysis/frame_vs_track_coverage.csv" ]]; finish 09_frame_track_coverage_gap; }
s10() { if ! begin 10_matched_only_model_recompute; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/end_to_end/matched_only_reference_model.csv" ]]; finish 10_matched_only_model_recompute; }
s11() { if ! begin 11_revised_bottleneck; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2_5/analysis/revised_bottleneck_summary.csv" ]]; finish 11_revised_bottleneck; }
s12() { if ! begin 12_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/iclr27_phase2_5/test_iclr27_phase2_5.py"; finish 12_unit_tests; }
s13() { if ! begin 13_gate; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2_5/REVISED_BOTTLENECK_CONCLUSION.md" ]]
  printf 'READY_TO_DUMP_PRE_ASSOC_DETECTIONS\n' > "$ROOT/runs/iclr27_phase2_5/status.txt"
  finish 13_gate; }
s14() { if ! begin 14_final_report; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2_5/PHASE2_5_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/iclr27_phase2_5/PHASE2_5_DECISION.md" ]]
  finish 14_final_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/iclr27_phase2_5/status.txt" 2>/dev/null || echo PENDING)"
