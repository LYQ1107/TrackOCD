#!/usr/bin/env bash
# Architecture 1.5 blocking pipeline (resumable).
# Stages: 00 preflight, 01 baseline audit, 02 B2 repro, 03-07 Stage A sweep,
# 08 Stage A gate, 09-10 track audit, 11-18 Stage B/C (implemented when gates
# allow). Each stage writes .launched/.done markers under runs/arch1_5/markers.
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
MK="$ROOT/runs/arch1_5/markers"
LOG="$ROOT/runs/arch1_5"
OUT="$ROOT/outputs/arch1_5"

mkdir -p "$MK" "$OUT/metrics" "$OUT/track_audit" "$OUT/tracking"

stage_ok() { [[ -f "$MK/$1.done" ]]; }

begin() {
  local name="$1"
  if stage_ok "$name"; then
    echo "[skip] $name (done marker exists)"
    return 1
  fi
  echo "[run ] $name"
  touch "$MK/$name.launched"
  return 0
}

finish() {
  local name="$1"
  local tmp="$MK/$name.done.tmp"
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$name" > "$tmp"
  mv "$tmp" "$MK/$name.done"
  echo "[done] $name"
}

export PYTHONPATH="$ROOT"

stage00_preflight() {
  if ! begin 00_preflight; then return 0; fi
  "$PY" - <<'PY'
import json, os, shutil, sys
root = "/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
checks = {
    "raw_tao": os.path.isdir("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao"),
    "symlink": os.path.islink(os.path.join(root, "data/raw/tao")),
    "val_predictions": os.path.exists(os.path.join(root, "outputs/simowt/val_predictions.json")),
    "gt_labels": os.path.exists(os.path.join(root, "data/tao_ow_ocd_v1/private/val_gt_track_labels.jsonl")),
    "b2_csv": os.path.exists(os.path.join(root, "outputs/arch1_5/metrics/b2_reproduced.csv")),
}
mem = os.popen("free -g").read().splitlines()[1].split()
out = {"checks": checks, "mem_free_gb": float(mem[3]), "mem_avail_gb": float(mem[6])}
open(os.path.join(root, "runs/arch1_5/preflight.json"), "w").write(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
if not all(checks.values()):
    sys.exit(1)
PY
  finish 00_preflight
}

stage01_baseline_audit() {
  if ! begin 01_baseline_audit; then return 0; fi
  [[ -f "$ROOT/docs/ARCH1_5_BASELINE_AUDIT.md" ]]
  [[ -f "$ROOT/docs/ARCH1_REPORT.md" ]]
  [[ -f "$ROOT/outputs/metrics/summary.csv" ]]
  finish 01_baseline_audit
}

stage02_reproduce_b2() {
  if ! begin 02_reproduce_b2; then return 0; fi
  "$PY" "$ROOT/src/ocd_v2/run_stage_a.py" --stage b2
  n=$(wc -l < "$OUT/metrics/b2_reproduced.csv")
  [[ "$n" -ge 5 ]]
  finish 02_reproduce_b2
}

stage03_proxy_gate_calibration() {
  if ! begin 03_proxy_gate_calibration; then return 0; fi
  # resume: stage work already completed if all outputs exist
  if [[ -f "$LOG/calibrated_params.json" && -f "$OUT/metrics/learned_gate.csv" ]]; then
    echo "[resume] 03 outputs already complete"
    finish 03_proxy_gate_calibration
    return 0
  fi
  "$PY" "$ROOT/src/ocd_v2/run_stage_a.py" --stage gates
  [[ -f "$OUT/metrics/learned_gate.csv" ]]
  [[ -f "$LOG/calibrated_params.json" ]]
  finish 03_proxy_gate_calibration
}

stage04_oracle_gate_diagnostics() {
  if ! begin 04_oracle_gate_diagnostics; then return 0; fi
  "$PY" "$ROOT/src/ocd_v2/run_stage_a.py" --stage oracle
  [[ -f "$OUT/metrics/oracle_gate.csv" ]]
  finish 04_oracle_gate_diagnostics
}

stage05_online_clustering_baselines() {
  if ! begin 05_online_clustering_baselines; then return 0; fi
  # repair/rebuild per-gate CSVs from per-run JSONs (idempotent)
  "$PY" "$ROOT/src/ocd_v2/rebuild_csvs.py"
  "$PY" "$ROOT/src/ocd_v2/summarize_stage_a.py" > "$LOG/stage_a_summarize.log" 2>&1
  for m in spherical_kmeans dpmeans candidate_buffer; do
    [[ -f "$OUT/metrics/${m}.csv" ]]
  done
  finish 05_online_clustering_baselines
}

stage06_ocd_v2_multicenter() {
  if ! begin 06_ocd_v2_multicenter; then return 0; fi
  [[ -f "$OUT/metrics/ocd_v2.csv" ]]
  finish 06_ocd_v2_multicenter
}

stage07_stage_a_evaluation() {
  if ! begin 07_stage_a_evaluation; then return 0; fi
  "$PY" "$ROOT/src/ocd_v2/summarize_stage_a.py" | tee "$LOG/stage_a_summarize.log"
  [[ -f "$OUT/metrics/stage_a_summary.csv" ]]
  finish 07_stage_a_evaluation
}

stage08_stage_a_gate() {
  if ! begin 08_stage_a_gate; then return 0; fi
  [[ -f "$LOG/stage_a_gate.json" ]]
  # resume: gate result already computed
  if [[ -f "$LOG/final_status.txt" ]]; then
    echo "[resume] 08 gate result exists"
    finish 08_stage_a_gate
    return 0
  fi
  "$PY" - "$LOG/stage_a_gate.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("STAGE_A_PASSED" if r["passed"] else "STAGE_A_FAILED")
print(json.dumps(r["checks"], indent=2))
open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/runs/arch1_5/final_status.txt", "w").write(
    "PASS_STAGE_A\n" if r["passed"] else "STOP_AFTER_STAGE_A\n"
)
PY
  finish 08_stage_a_gate
}

stage09_track_output_audit() {
  if ! begin 09_track_output_audit; then return 0; fi
  avail=$(free -g | awk '/Mem:/{print $7}')
  if [[ "$avail" -lt 40 ]]; then
    echo "ABORT 09: only ${avail}GB available, need >=40GB for 1.2GB JSON load"
    return 1
  fi
  "$PY" "$ROOT/src/track_stabilization/audit.py" --stage basic
  [[ -f "$OUT/track_audit/pred_track_stats.json" ]]
  finish 09_track_output_audit
}

stage10_track_stats() {
  if ! begin 10_track_stats; then return 0; fi
  "$PY" "$ROOT/src/track_stabilization/audit.py" --stage gtfrag
  [[ -f "$OUT/track_audit/gt_fragmentation_stats.csv" ]]
  finish 10_track_stats
}

stage11_stitching_calibration() {
  if stage_ok 08_stage_a_gate && ! grep -q PASS "$LOG/stage_a_gate.json" 2>/dev/null; then
    echo "Stage A failed; stitching not authorized."
    return 0
  fi
  if ! begin 11_stitching_calibration; then return 0; fi
  echo "Stage 11 not implemented yet (needs Stage A pass + audit results); stopping here."
  printf 'PASS_STAGE_A_PENDING_STAGE_B_IMPL\n' > "$LOG/final_status.txt"
  finish 11_stitching_calibration
}

main() {
  stage00_preflight
  stage01_baseline_audit
  stage02_reproduce_b2
  stage03_proxy_gate_calibration
  stage04_oracle_gate_diagnostics
  stage05_online_clustering_baselines
  stage06_ocd_v2_multicenter
  stage07_stage_a_evaluation
  stage08_stage_a_gate
  stage09_track_output_audit
  stage10_track_stats
  stage11_stitching_calibration
  echo "PIPELINE_FINISHED status=$(cat "$LOG/final_status.txt" 2>/dev/null || echo UNKNOWN)"
}

main
