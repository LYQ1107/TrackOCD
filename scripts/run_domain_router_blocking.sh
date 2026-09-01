#!/usr/bin/env bash
# Domain-Robust Router (Stage R) blocking pipeline with conditional Stage E.
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/domain_router/markers"
mkdir -p "$MK" "$ROOT/runs/domain_router" "$ROOT/outputs/domain_router/metrics" "$ROOT/outputs/domain_router/audit" "$ROOT/outputs/domain_router/tests" "$ROOT/data/domain_router"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
skipped() { touch "$MK/$1.skipped"; echo "[skipped] $1"; }

gate_status() {
  "$PY" -c "import json;print(json.load(open('$ROOT/runs/domain_router/router_gate.json'))['status'])"
}

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/data/trackocd_v1/manifests/manifest_v1.0.json" ]]
  finish 00_preflight; }
s01() { if ! begin 01_baseline_audit; then return 0; fi
  [[ -f "$ROOT/docs/domain_router/BASELINE_AND_FAILURE_AUDIT.md" ]]; finish 01_baseline_audit; }
s02() { if ! begin 02_domain_metadata_audit; then return 0; fi
  [[ -f "$ROOT/docs/domain_router/DOMAIN_METADATA_AUDIT.md" ]]
  [[ -f "$ROOT/outputs/domain_router/audit/domain_metadata.csv" ]]; finish 02_domain_metadata_audit; }
s03() { if ! begin 03_build_domain_proxy; then return 0; fi
  [[ -f "$ROOT/data/domain_router/proxy_protocol/manifest.json" ]]; finish 03_build_domain_proxy; }
s04() { if ! begin 04_freeze_proxy_manifest; then return 0; fi
  [[ -d "$ROOT/data/domain_router/proxy_protocol/hashes" ]]; finish 04_freeze_proxy_manifest; }
s05() { if ! begin 05_implement_router_features; then return 0; fi
  [[ -f "$ROOT/src/domain_router/features/router_features.py" ]]; finish 05_implement_router_features; }
s06() { if ! begin 06_router_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/domain_router/test_domain_router.py"; finish 06_router_unit_tests; }
s07() { if ! begin 07_reproduce_r0; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/r0_reproduction.csv" ]]; finish 07_reproduce_r0; }
s08() { if ! begin 08_nested_proxy_r1; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/proxy_outer_folds.csv" ]]; finish 08_nested_proxy_r1; }
s09() { s08; }; s10() { s08; }; s11() { s08; }
s12() { if ! begin 12_conditional_r5; then return 0; fi
  if [[ -f "$ROOT/runs/domain_router/r5.skipped" ]]; then skipped 12_conditional_r5; else finish 12_conditional_r5; fi; }
s13() { if ! begin 13_select_router_train_only; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/router_selection.csv" ]]; finish 13_select_router_train_only; }
s14() { if ! begin 14_run_router_full_eval; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/router_full_results.csv" ]]; finish 14_run_router_full_eval; }
s15() { if ! begin 15_router_subgroup_diagnostics; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/router_subgroups.csv" ]]; finish 15_router_subgroup_diagnostics; }
s16() { if ! begin 16_router_gate; then return 0; fi
  [[ -f "$ROOT/runs/domain_router/router_gate.json" ]]; finish 16_router_gate; }
s17() { if stage_ok 17_conditional_encoder_setup; then return 0; fi
  if [[ "$(gate_status)" != "PASS_DOMAIN_ROBUST_ROUTER" ]]; then skipped 17_conditional_encoder_setup; return 0; fi
  finish 17_conditional_encoder_setup; }
s18() { s17; }; s19() { s17; }; s20() { s17; }; s21() { s17; }; s22() { s17; }; s23() { s17; }
s24() { if ! begin 24_final_analysis; then return 0; fi
  [[ -f "$ROOT/outputs/domain_router/metrics/router_final_summary.csv" ]]; finish 24_final_analysis; }
s25() { if ! begin 25_report; then return 0; fi
  [[ -f "$ROOT/docs/domain_router/DOMAIN_ROUTER_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/domain_router/DOMAIN_ROUTER_DECISION.md" ]]
  printf '%s\n' "$(gate_status)" > "$ROOT/runs/domain_router/status.txt"
  finish 25_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14; s15; s16
s17; s18; s19; s20; s21; s22; s23; s24; s25
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/domain_router/status.txt" 2>/dev/null || echo PENDING)"
