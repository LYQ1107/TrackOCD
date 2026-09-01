#!/usr/bin/env bash
# TrackOCD Hard Dual-Branch blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/dual_branch/markers"
mkdir -p "$MK" "$ROOT/outputs/dual_branch/metrics" "$ROOT/outputs/dual_branch/tests" "$ROOT/runs/dual_branch"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  "$PY" - <<'PY'
import os
r="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
assert os.path.exists(r+"/runs/trackocd_v1/traj_enc_transformer/model.pth")
assert os.path.exists(r+"/data/trackocd_v1/manifests/manifest_v1.0.json")
PY
  finish 00_preflight; }
s01() { if ! begin 01_baseline_audit; then return 0; fi
  [[ -f "$ROOT/docs/dual_branch/BASELINE_AUDIT.md" ]]; finish 01_baseline_audit; }
s02() { if ! begin 02_clone_reference_repos; then return 0; fi
  for d in SimGCD GET OCGCD hilo; do [[ -d "$ROOT/third_party/reference/$d/.git" ]]; done
  [[ -f "$ROOT/third_party/reference/REPOSITIONS.md" ]]; finish 02_clone_reference_repos; }
s03() { if ! begin 03_reference_code_review; then return 0; fi
  [[ -f "$ROOT/docs/dual_branch/REFERENCE_CODE_REVIEW.md" ]]; finish 03_reference_code_review; }
s04() { if ! begin 04_dual_branch_design; then return 0; fi
  [[ -f "$ROOT/docs/dual_branch/DUAL_BRANCH_DESIGN.md" ]]; finish 04_dual_branch_design; }
s05() { if ! begin 05_implementation; then return 0; fi
  [[ -f "$ROOT/src/dual_branch/evaluation/run_dual_branch.py" ]]; finish 05_implementation; }
s06() { if ! begin 06_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/dual_branch/test_dual_branch.py"; finish 06_unit_tests; }
s07() { if ! begin 07_reproduce_d0; then return 0; fi
  [[ -f "$ROOT/outputs/dual_branch/metrics/d0_reproduction.csv" ]]; finish 07_reproduce_d0; }
s08() { if ! begin 08_reproduce_d1; then return 0; fi
  [[ -f "$ROOT/outputs/dual_branch/metrics/d1_reproduction.csv" ]]; finish 08_reproduce_d1; }
s09() { if ! begin 09_run_d2; then return 0; fi
  [[ -f "$ROOT/outputs/dual_branch/metrics/d2_hard_dual_branch.csv" ]]; finish 09_run_d2; }
s10() { if ! begin 10_run_d3_oracle_diagnostic; then return 0; fi
  [[ -f "$ROOT/outputs/dual_branch/metrics/d3_oracle_route.csv" ]]; finish 10_run_d3_oracle_diagnostic; }
s11() { if ! begin 11_paired_route_analysis; then return 0; fi
  [[ -f "$ROOT/outputs/dual_branch/metrics/paired_route_diagnostics.csv" ]]; finish 11_paired_route_analysis; }
s12() { if ! begin 12_stage_gate; then return 0; fi
  "$PY" "$ROOT/src/dual_branch/evaluation/summarize_dual_branch.py" > /dev/null
  [[ -f "$ROOT/runs/dual_branch/dual_branch_gate.json" ]]; finish 12_stage_gate; }
s13() { if ! begin 13_optional_d4; then return 0; fi
  [[ "$(cat "$ROOT/runs/dual_branch/status.txt")" == "PARTIAL_DUAL_BRANCH" ]] || echo "D4 not authorized (status=$(cat "$ROOT/runs/dual_branch/status.txt"))"
  finish 13_optional_d4; }
s14() { if ! begin 14_final_evaluation; then return 0; fi
  "$PY" "$ROOT/src/dual_branch/evaluation/summarize_dual_branch.py" > /dev/null; finish 14_final_evaluation; }
s15() { if ! begin 15_report; then return 0; fi
  [[ -f "$ROOT/docs/dual_branch/DUAL_BRANCH_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/dual_branch/DUAL_BRANCH_DECISION.md" ]]; finish 15_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14; s15
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/dual_branch/status.txt")"
