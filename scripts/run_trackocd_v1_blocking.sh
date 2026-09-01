#!/usr/bin/env bash
# TrackOCD-v1.0 blocking pipeline (resumable).
# Stages: 00 preflight, 01 legacy audit, 02 pure protocol, 03 ov protocol,
# 04 public/private, 05 evaluator, 06 unit tests, 07 freeze, 08 baselines,
# 09 legacy-vs-corrected, 10 modular, 11 trajectory train, 12 trajectory eval,
# 13 stitching, 14 bidirectional, 15 bakeoff, 16 report.
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/trackocd_v1/markers"
LOG="$ROOT/runs/trackocd_v1"
OUT="$ROOT/outputs/trackocd_v1"

mkdir -p "$MK" "$OUT/metrics" "$OUT/tests" "$OUT/baselines"

stage_ok() { [[ -f "$MK/$1.done" ]]; }

begin() {
  if stage_ok "$1"; then echo "[skip] $1"; return 1; fi
  echo "[run ] $1"
  touch "$MK/$1.launched"
  return 0
}

finish() {
  echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"
  mv "$MK/$1.done.tmp" "$MK/$1.done"
  echo "[done] $1"
}

export PYTHONPATH="$ROOT"

stage00() {
  if ! begin 00_preflight; then return 0; fi
  "$PY" - <<'PY'
import json, os
root="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
ok = all(os.path.exists(os.path.join(root, p)) for p in [
  "data/raw/tao/annotations/train.json",
  "data/raw/tao/annotations/validation.json",
  "outputs/simowt/val_predictions.json",
  "outputs/arch1_5/track_audit/pred_tracks_compact.pkl"])
json.dump({"ok": ok}, open(os.path.join(root, "runs/trackocd_v1/preflight.json"), "w"))
assert ok
PY
  finish 00_preflight
}

stage01() {
  if ! begin 01_legacy_audit; then return 0; fi
  [[ -f "$ROOT/docs/trackocd_v1/LEGACY_RESULT_AUDIT.md" ]]
  finish 01_legacy_audit
}

stage02() {
  if ! begin 02_build_pure_protocol; then return 0; fi
  [[ -f "$ROOT/data/trackocd_v1/pure/private/val_gt_track_labels.jsonl" ]]
  finish 02_build_pure_protocol
}

stage03() {
  if ! begin 03_build_ov_assisted_protocol; then return 0; fi
  [[ -f "$ROOT/data/trackocd_v1/ov_assisted/private/val_gt_track_labels.jsonl" ]]
  finish 03_build_ov_assisted_protocol
}

stage04() {
  if ! begin 04_rebuild_public_private_data; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/build_protocols.py"
  [[ -f "$ROOT/data/trackocd_v1/protocols.json" ]]
  finish 04_rebuild_public_private_data
}

stage05() {
  if ! begin 05_fix_evaluator; then return 0; fi
  [[ -f "$ROOT/src/trackocd_v1/evaluation/trackocd_evaluator.py" ]]
  finish 05_fix_evaluator
}

stage06() {
  if ! begin 06_evaluator_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/test_trackocd_evaluator.py"
  finish 06_evaluator_unit_tests
}

stage07() {
  if ! begin 07_freeze_v1_manifests; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/freeze_v1.py"
  [[ -f "$ROOT/data/trackocd_v1/manifests/manifest_v1.0.json" ]]
  finish 07_freeze_v1_manifests
}

stage08() {
  if ! begin 08_rerun_corrected_baselines; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/rerun_baselines.py"
  [[ -f "$OUT/metrics/pure_baselines.csv" && -f "$OUT/metrics/ov_assisted_baselines.csv" ]]
  finish 08_rerun_corrected_baselines
}

stage09() {
  if ! begin 09_legacy_corrected_comparison; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/summarize_baselines.py"
  [[ -f "$OUT/metrics/corrected_baseline_summary.csv" && -f "$OUT/metrics/legacy_vs_corrected.csv" ]]
  finish 09_legacy_corrected_comparison
}

stage10() {
  if ! begin 10_modular_architecture; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/modular.py"
  [[ -f "$OUT/metrics/modular_architecture.csv" ]]
  finish 10_modular_architecture
}

stage11() {
  if ! begin 11_train_trajectory_encoder; then return 0; fi
  CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/src/trackocd_v1/trajectory_encoder.py" --mode train --variant attn_pool
  CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/src/trackocd_v1/trajectory_encoder.py" --mode train --variant transformer
  [[ -f "$LOG/traj_enc_attn_pool/model.pth" && -f "$LOG/traj_enc_transformer/model.pth" ]]
  finish 11_train_trajectory_encoder
}

stage12() {
  if ! begin 12_trajectory_architecture; then return 0; fi
  CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/src/trackocd_v1/trajectory_encoder.py" --mode eval --variant attn_pool
  CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/src/trackocd_v1/trajectory_encoder.py" --mode eval --variant transformer
  "$PY" - <<'PY'
import csv
rows=[]
for v in ("attn_pool","transformer"):
    with open(f"/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/trackocd_v1/metrics/trajectory_architecture_{v}.csv") as f:
        rows.extend(csv.DictReader(f))
with open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/trackocd_v1/metrics/trajectory_architecture.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
PY
  [[ -f "$OUT/metrics/trajectory_architecture.csv" ]]
  finish 12_trajectory_architecture
}

stage13() {
  if ! begin 13_stitching_baseline; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/stitching.py" --max-gap 30
  [[ -f "$OUT/metrics/bidirectional_feasibility_c0.json" ]]
  finish 13_stitching_baseline
}

stage14() {
  if ! begin 14_bidirectional_feasibility; then return 0; fi
  "$PY" "$ROOT/src/trackocd_v1/stitching.py" --c1c2 --max-gap 30 --identity-thr 0.60
  [[ -f "$OUT/metrics/bidirectional_feasibility_c1c2.json" ]]
  finish 14_bidirectional_feasibility
}

stage15() {
  if ! begin 15_architecture_bakeoff; then return 0; fi
  CUDA_VISIBLE_DEVICES=4 "$PY" "$ROOT/src/trackocd_v1/bakeoff.py"
  [[ -f "$OUT/metrics/architecture_bakeoff.csv" ]]
  finish 15_architecture_bakeoff
}

stage16() {
  if ! begin 16_final_report; then return 0; fi
  [[ -f "$ROOT/docs/trackocd_v1/TRACKOCD_V1_FINAL_REPORT.md" ]]
  [[ -f "$ROOT/docs/trackocd_v1/ARCHITECTURE_DECISION.md" ]]
  printf 'PASS_BENCHMARK_AND_BAKEOFF\n' > "$LOG/status.txt"
  finish 16_final_report
}

main() {
  stage00; stage01; stage02; stage03; stage04; stage05; stage06; stage07
  stage08; stage09; stage10; stage11; stage12; stage13; stage14; stage15; stage16
  echo "PIPELINE_FINISHED status=$(cat "$LOG/status.txt")"
}

main
