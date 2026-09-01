#!/usr/bin/env bash
# ICLR27 Phase 2 blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
PY37="/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python"
MK="$ROOT/runs/iclr27_phase2/markers"
mkdir -p "$MK" "$ROOT/runs/iclr27_phase2" "$ROOT/outputs/iclr27_phase2/audit" \
  "$ROOT/outputs/iclr27_phase2/tracking/simowt" "$ROOT/outputs/iclr27_phase2/end_to_end" \
  "$ROOT/outputs/iclr27_phase2/analysis" "$ROOT/outputs/iclr27_phase2/tests"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
skipped() { touch "$MK/$1.skipped"; echo "[skipped] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/outputs/simowt/val_predictions.json" ]]; finish 00_preflight; }
s01() { if ! begin 01_read_agents; then return 0; fi
  [[ -f "$ROOT/AGENTS.md" ]]; finish 01_read_agents; }
s02() { if ! begin 02_read_phase1; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_closure/ICLR27_PHASE1_FINAL_REPORT.md" ]]; finish 02_read_phase1; }
s03() { if ! begin 03_hash_inputs; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/audit/input_hashes.json" ]]; finish 03_hash_inputs; }
s04() { if ! begin 04_detection_file_search; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/audit/detection_candidate_inventory.csv" ]]; finish 04_detection_file_search; }
s05() { if ! begin 05_detection_content_audit; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2/DETECTION_PROVENANCE_AUDIT.md" ]]; finish 05_detection_content_audit; }
s06() { if ! begin 06_detection_provenance_decision; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/audit/detection_provenance_decision.json" ]]; finish 06_detection_provenance_decision; }
s07() { if ! begin 07_trackeval_adapter_audit; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2/TRACKEVAL_ADAPTER_EXTENSION.md" ]]; finish 07_trackeval_adapter_audit; }
s08() { if ! begin 08_extend_clear_identity; then return 0; fi
  for s in all known unknown; do
    [[ -f "$ROOT/outputs/iclr27_phase2/tracking/simowt/clear_identity_$s/combined.json" ]]
  done
  finish 08_extend_clear_identity; }
s09() { if ! begin 09_simowt_re_evaluation; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/tracking/simowt/summary.csv" ]]; finish 09_simowt_re_evaluation; }
s10() { if ! begin 10_fragmentation_definition_audit; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/tracking/fragmentation_definition_comparison.csv" ]]; finish 10_fragmentation_definition_audit; }
s11() { if ! begin 11_matched_only_known_audit; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/audit/matched_only_known_reconstruction.csv" ]]; finish 11_matched_only_known_audit; }
s12() { if ! begin 12_branch_gate; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/audit/detection_provenance_decision.json" ]]
  finish 12_branch_gate; }
s13a() { if ! begin 13a_reproduce_detector_smoke; then return 0; fi
  skipped 13a_reproduce_detector_smoke; }
s13b() { if ! begin 13b_extract_full_detections; then return 0; fi
  skipped 13b_extract_full_detections; }
s14() { if ! begin 14_bytetrack_smoke; then return 0; fi
  skipped 14_bytetrack_smoke; }
s15() { if ! begin 15_bytetrack_full; then return 0; fi
  skipped 15_bytetrack_full; }
s16() { if ! begin 16_second_frontend_trackeval; then return 0; fi
  skipped 16_second_frontend_trackeval; }
s17() { if ! begin 17_second_frontend_feature_build; then return 0; fi
  skipped 17_second_frontend_feature_build; }
s18() { if ! begin 18_trackocd_reference_evaluation; then return 0; fi
  skipped 18_trackocd_reference_evaluation; }
s19() { if ! begin 19_coverage_aware_evaluation; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/end_to_end/simowt_results.csv" ]]; finish 19_coverage_aware_evaluation; }
s20() { if ! begin 20_bottleneck_decomposition; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase2/analysis/bottleneck_decomposition.csv" ]]; finish 20_bottleneck_decomposition; }
s21() { if ! begin 21_claim_evidence_update; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2/PHASE2_FINAL_REPORT.md" ]]; finish 21_claim_evidence_update; }
s22() { if ! begin 22_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/iclr27_phase2/test_iclr27_phase2.py"; finish 22_unit_tests; }
s23() { if ! begin 23_final_report; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2/PHASE2_DECISION.md" ]]
  printf 'CONTROLLED_SECOND_FRONTEND_BLOCKED\n' > "$ROOT/runs/iclr27_phase2/status.txt"
  finish 23_final_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12
s13a; s13b; s14; s15; s16; s17; s18; s19; s20; s21; s22; s23
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/iclr27_phase2/status.txt" 2>/dev/null || echo PENDING)"
