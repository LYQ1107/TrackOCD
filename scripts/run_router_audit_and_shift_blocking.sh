#!/usr/bin/env bash
# Router audit (Part A) + conditional causal score-shift (Part B) pipeline.
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/router_audit_and_shift/markers"
mkdir -p "$MK" "$ROOT/runs/router_audit" "$ROOT/runs/causal_score_shift" \
  "$ROOT/outputs/router_audit/tests" "$ROOT/outputs/causal_score_shift/metrics" \
  "$ROOT/outputs/causal_score_shift/tests"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
skipped() { touch "$MK/$1.skipped"; echo "[skipped] $1"; }

audit_status() { "$PY" -c "import json;print(json.load(open('$ROOT/runs/router_audit/audit_gate.json'))['status'])"; }

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/data/trackocd_v1/manifests/manifest_v1.0.json" ]]; finish 00_preflight; }
s01() { if ! begin 01_read_existing_artifacts; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/router_full_results.csv" ]]; finish 01_read_existing_artifacts; }
s02() { if ! begin 02_hash_audit_inputs; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/audit_input_hashes.json" ]]; finish 02_hash_audit_inputs; }
s03() { if ! begin 03_method_registry_audit; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/method_registry_audit.csv" ]]; finish 03_method_registry_audit; }
s04() { if ! begin 04_r1_result_reconstruction; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/r1_result_reconstruction.csv" ]]; finish 04_r1_result_reconstruction; }
s05() { if ! begin 05_selection_reconstruction; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/selection_reconstruction.csv" ]]; finish 05_selection_reconstruction; }
s06() { if ! begin 06_fold_feasibility_audit; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/fold_feasibility.csv" ]]; finish 06_fold_feasibility_audit; }
s07() { if ! begin 07_threshold_aggregation_audit; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/threshold_aggregation_comparison.csv" ]]; finish 07_threshold_aggregation_audit; }
s08() { if ! begin 08_repair_cycle_audit; then return 0; fi
  [[ -f "$ROOT/docs/router_audit/REPAIR_CYCLE_AUDIT.md" ]]; finish 08_repair_cycle_audit; }
s09() { if ! begin 09_audit_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/router_audit/test_router_audit.py"; finish 09_audit_unit_tests; }
s10() { if ! begin 10_minimal_corrections; then return 0; fi
  "$PY" "$ROOT/src/router_audit/finalize_selection.py"; finish 10_minimal_corrections; }
s11() { if ! begin 11_recompute_corrected_router_results; then return 0; fi
  [[ -f "$ROOT/outputs/router_audit/corrected_router_full_results.csv" ]]; finish 11_recompute_corrected_router_results; }
s12() { if ! begin 12_audit_gate; then return 0; fi
  [[ -f "$ROOT/runs/router_audit/audit_gate.json" ]]; finish 12_audit_gate; }
s13() { if ! begin 13_freeze_audit_report; then return 0; fi
  [[ -f "$ROOT/docs/router_audit/ROUTER_AUDIT_FINAL_REPORT.md" ]]; finish 13_freeze_audit_report; }

s14() { if stage_ok 14_conditional_build_causal_stream; then return 0; fi
  if [[ "$(audit_status)" != "AUDIT_CONFIRMED_NO_GAIN" && "$(audit_status)" != "AUDIT_CORRECTION_STILL_NO_GAIN" ]]; then
    skipped 14_conditional_build_causal_stream; return 0
  fi
  finish 14_conditional_build_causal_stream; }
s15() { s14; }; s16() { s14; }; s17() { s14; }; s18() { s14; }
s19() { if ! begin 19_conditional_causal_unit_tests; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 19_conditional_causal_unit_tests; return 0; }
  "$PY" "$ROOT/tests/causal_score_shift/test_causal_score_shift.py"; finish 19_conditional_causal_unit_tests; }
s20() { if ! begin 20_conditional_train_proxy_simulation; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 20_conditional_train_proxy_simulation; return 0; }
  [[ -f "$ROOT/outputs/causal_score_shift/metrics/train_proxy_results.csv" ]]; finish 20_conditional_train_proxy_simulation; }
s21() { if ! begin 21_conditional_stress_tests; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 21_conditional_stress_tests; return 0; }
  [[ -f "$ROOT/outputs/causal_score_shift/metrics/stress_tests.csv" ]]; finish 21_conditional_stress_tests; }
s22() { if ! begin 22_conditional_select_method; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 22_conditional_select_method; return 0; }
  [[ -f "$ROOT/outputs/causal_score_shift/metrics/method_selection.csv" ]]; finish 22_conditional_select_method; }
s23() { if ! begin 23_conditional_full_evaluation; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 23_conditional_full_evaluation; return 0; }
  [[ -f "$ROOT/outputs/causal_score_shift/metrics/full_results.csv" ]]; finish 23_conditional_full_evaluation; }
s24() { if ! begin 24_conditional_subgroup_analysis; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 24_conditional_subgroup_analysis; return 0; }
  [[ -f "$ROOT/outputs/causal_score_shift/metrics/subgroup_results.csv" ]]; finish 24_conditional_subgroup_analysis; }
s25() { if ! begin 25_conditional_causal_gate; then return 0; fi
  [[ "$(audit_status)" == "AUDIT_CONFIRMED_NO_GAIN" ]] || { skipped 25_conditional_causal_gate; return 0; }
  [[ -f "$ROOT/runs/causal_score_shift/causal_gate.json" ]]; finish 25_conditional_causal_gate; }
s26() { if ! begin 26_final_report; then return 0; fi
  [[ -f "$ROOT/docs/causal_score_shift/CAUSAL_SCORE_SHIFT_FINAL_REPORT.md" ]]
  printf '%s\n' "$(audit_status)" > "$ROOT/runs/router_audit_and_shift/status.txt"
  finish 26_final_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13
s14; s15; s16; s17; s18; s19; s20; s21; s22; s23; s24; s25; s26
echo "PIPELINE_FINISHED"
