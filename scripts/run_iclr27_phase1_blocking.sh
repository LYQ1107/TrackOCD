#!/usr/bin/env bash
# ICLR27 Phase 1 blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/iclr27_closure/markers"
mkdir -p "$MK" "$ROOT/runs/iclr27_closure" "$ROOT/outputs/iclr27_closure/audit" \
  "$ROOT/outputs/iclr27_closure/tables" "$ROOT/outputs/iclr27_closure/end_to_end" \
  "$ROOT/outputs/iclr27_closure/planning" "$ROOT/outputs/iclr27_closure/tests"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
skipped() { touch "$MK/$1.skipped"; echo "[skipped] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/outputs/simowt/val_predictions.json" ]]; finish 00_preflight; }
s01() { if ! begin 01_read_agents; then return 0; fi
  [[ -f "$ROOT/AGENTS.md" ]]; finish 01_read_agents; }
s02() { if ! begin 02_artifact_inventory; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/audit/artifact_inventory.csv" ]]; finish 02_artifact_inventory; }
s03() { if ! begin 03_hash_existing_artifacts; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/audit/artifact_hashes.json" ]]; finish 03_hash_existing_artifacts; }
s04() { if ! begin 04_freeze_positioning; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_closure/PAPER_POSITIONING.md" ]]; finish 04_freeze_positioning; }
s05() { if ! begin 05_freeze_naming; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/audit/method_name_mapping.csv" ]]; finish 05_freeze_naming; }
s06() { if ! begin 06_freeze_protocol; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tables/protocol_statistics.csv" ]]; finish 06_freeze_protocol; }
s07() { if ! begin 07_evaluator_semantics_audit; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tables/legacy_vs_corrected_metrics.csv" ]]; finish 07_evaluator_semantics_audit; }
s08() { if ! begin 08_build_toy_evaluator_case; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/figures/evaluator_toy_example.json" ]]; finish 08_build_toy_evaluator_case; }
s09() { if ! begin 09_tracking_frontend_audit; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_closure/TRACKING_FRONTEND_AUDIT.md" ]]; finish 09_tracking_frontend_audit; }
s10() { if ! begin 10_tracking_conversion_tests; then return 0; fi
  [[ -f "$ROOT/third_party/TrackEval/data/trackers/tao/tao_validation/simowt/data/predictions.json" ]]; finish 10_tracking_conversion_tests; }
s11() { if ! begin 11_run_trackeval; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tracking_eval/simowt/summary.csv" ]]; finish 11_run_trackeval; }
s12() { if ! begin 12_role_stratified_tracking; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tracking_eval/simowt/known_role_metrics.csv" ]]; finish 12_role_stratified_tracking; }
s13() { if ! begin 13_track_coverage_analysis; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tables/track_coverage_table.csv" ]]; finish 13_track_coverage_analysis; }
s14() { if ! begin 14_predicted_track_matched_only; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/end_to_end/matched_only_results.csv" ]]; finish 14_predicted_track_matched_only; }
s15() { if ! begin 15_predicted_track_coverage_aware; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/end_to_end/coverage_aware_results.csv" ]]; finish 15_predicted_track_coverage_aware; }
s16() { if ! begin 16_rebuild_gt_track_tables; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/tables/gt_track_main_table.csv" ]]; finish 16_rebuild_gt_track_tables; }
s17() { if ! begin 17_claim_evidence_matrix; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/audit/claim_evidence_matrix.csv" ]]; finish 17_claim_evidence_matrix; }
s18() { if ! begin 18_review_risk_register; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_closure/ICLR_REVIEW_RISK_REGISTER.md" ]]; finish 18_review_risk_register; }
s19() { if ! begin 19_external_baseline_shortlist; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/planning/external_baselines.csv" ]]; finish 19_external_baseline_shortlist; }
s20() { if ! begin 20_second_tracker_shortlist; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/planning/tracker_candidates.csv" ]]; finish 20_second_tracker_shortlist; }
s21() { if ! begin 21_freeze_experiment_matrix; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_closure/planning/frozen_experiment_matrix.csv" ]]; finish 21_freeze_experiment_matrix; }
s22() { if ! begin 22_build_paper_table_data; then return 0; fi
  "$PY" "$ROOT/scripts/build_paper_tables.py"; finish 22_build_paper_table_data; }
s23() { if ! begin 23_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/iclr27_closure/test_iclr27_closure.py"; finish 23_unit_tests; }
s24() { if ! begin 24_final_report; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_closure/ICLR27_PHASE1_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/iclr27_closure/ICLR27_READINESS_DECISION.md" ]]
  printf 'READY_FOR_PHASE2\n' > "$ROOT/runs/iclr27_closure/status.txt"
  finish 24_final_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14
s15; s16; s17; s18; s19; s20; s21; s22; s23; s24
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/iclr27_closure/status.txt" 2>/dev/null || echo PENDING)"
