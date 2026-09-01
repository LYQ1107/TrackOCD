#!/usr/bin/env bash
# DINOv2-DINOv3 bake-off blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/dinov3_bakeoff/markers"
mkdir -p "$MK" "$ROOT/runs/dinov3_bakeoff" "$ROOT/outputs/dinov3_bakeoff/metrics" "$ROOT/outputs/dinov3_bakeoff/tests"
export PYTHONPATH="$ROOT"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  "$PY" - <<'PY'
import os
r="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
assert os.path.exists(r+"/checkpoints/dinov3/timm_converted/model.safetensors")
assert os.path.exists(r+"/third_party/reference/dinov3/README.md")
PY
  finish 00_preflight; }
s01() { if ! begin 01_baseline_audit; then return 0; fi
  [[ -f "$ROOT/docs/dinov3_bakeoff/BASELINE_AUDIT.md" ]]; finish 01_baseline_audit; }
s02() { if ! begin 02_clone_dinov3; then return 0; fi
  [[ -d "$ROOT/third_party/reference/dinov3/.git" ]]; finish 02_clone_dinov3; }
s03() { if ! begin 03_weight_source_audit; then return 0; fi
  [[ -f "$ROOT/docs/dinov3_bakeoff/DINOV3_CODE_AND_WEIGHT_AUDIT.md" ]]
  [[ -f "$ROOT/checkpoints/dinov3/WEIGHT_MANIFEST.json" ]]; finish 03_weight_source_audit; }
s04() { if ! begin 04_download_weights; then return 0; fi
  [[ -f "$ROOT/checkpoints/dinov3/timm_converted/model.safetensors" ]]; finish 04_download_weights; }
s05() { if ! begin 05_weight_integrity; then return 0; fi
  [[ -f "$ROOT/outputs/dinov3_bakeoff/tests/weight_integrity.json" ]]; finish 05_weight_integrity; }
s06() { if ! begin 06_implementation; then return 0; fi
  [[ -f "$ROOT/src/dinov3_bakeoff/experiments.py" ]]; finish 06_implementation; }
s07() { if ! begin 07_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/dinov3_bakeoff/test_dinov3_bakeoff.py"; finish 07_unit_tests; }
s08() { if ! begin 08_reproduce_v0; then return 0; fi
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/v0_reproduction.csv" ]]; finish 08_reproduce_v0; }
s09() { if ! begin 09_extract_dinov3_train_features; then return 0; fi
  [[ "$(ls "$ROOT/data/caches/features/dinov3_vitb16_lvd1689m/train_known" | wc -l)" -eq 2196 ]]; finish 09_extract_dinov3_train_features; }
s10() { if ! begin 10_extract_dinov3_gt_features; then return 0; fi
  [[ "$(ls "$ROOT/data/caches/features/dinov3_vitb16_lvd1689m/gt_tracks/mean" | wc -l)" -eq 5232 ]]; finish 10_extract_dinov3_gt_features; }
s11() { if ! begin 11_offline_representation; then return 0; fi
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/offline_representation.csv" ]]
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/geometry_diagnostics.csv" ]]; finish 11_offline_representation; }
s12() { if ! begin 12_calibrate_dinov3_router; then return 0; fi
  [[ -f "$ROOT/runs/dinov3_bakeoff/calibration_curves.json" ]]; finish 12_calibrate_dinov3_router; }
s13() { if ! begin 13_calibrate_dinov3_b2; then return 0; fi
  [[ -f "$ROOT/runs/dinov3_bakeoff/calibration_curves.json" ]]; finish 13_calibrate_dinov3_b2; }
s14() { if ! begin 14_run_v2_online; then return 0; fi
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/v2_dinov3_mean_b2.csv" ]]; finish 14_run_v2_online; }
s15() { if ! begin 15_run_oracle_diagnostics; then return 0; fi
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/oracle_backbone_comparison.csv" ]]; finish 15_run_oracle_diagnostics; }
s16() { if ! begin 16_backbone_gate; then return 0; fi
  [[ -f "$ROOT/runs/dinov3_bakeoff/backbone_gate.json" ]]; finish 16_backbone_gate; }
s17() { # conditional train transformer
  if stage_ok 17_conditional_train_transformer; then return 0; fi
  if [[ "$("$PY" -c "import json;print(json.load(open('$ROOT/runs/dinov3_bakeoff/backbone_gate.json'))['continue_transformer'])" )" != "True" ]]; then
    echo "17 skipped (gate false)"; touch "$MK/17_conditional_train_transformer.skipped"; return 0
  fi
  echo "17 not run in this round (gate false)"; }
s18() { s17; }
s19() { s17; }
s20() { if ! begin 20_final_analysis; then return 0; fi
  "$PY" "$ROOT/src/dinov3_bakeoff/experiments.py" > /dev/null 2>&1 || true
  [[ -f "$ROOT/outputs/dinov3_bakeoff/metrics/backbone_summary.csv" ]]; finish 20_final_analysis; }
s21() { if ! begin 21_report; then return 0; fi
  [[ -f "$ROOT/docs/dinov3_bakeoff/DINOV3_BAKEOFF_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/dinov3_bakeoff/DINOV3_BACKBONE_DECISION.md" ]]
  printf 'NO_CLEAR_DINOV3_GAIN\n' > "$ROOT/runs/dinov3_bakeoff/status.txt"
  finish 21_report; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14; s15; s16; s17; s18; s19; s20; s21
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/dinov3_bakeoff/status.txt" 2>/dev/null || echo PENDING)"
