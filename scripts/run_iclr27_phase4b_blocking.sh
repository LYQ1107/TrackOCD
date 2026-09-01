#!/usr/bin/env bash
# Phase 4B blocking supervisor (reference; stages marked done from execution).
set -u
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
MK="$ROOT/runs/iclr27_phase4b/markers"
mkdir -p "$MK"
stage_done() { [[ -f "$MK/$1.done" ]]; }
if stage_done 01_update_agents; then echo "[skip] 01_update_agents"; fi
if stage_done 04_hota_field_audit; then echo "[skip] 04_hota_field_audit"; fi
if stage_done 08_bytetrack_feature_extraction; then echo "[skip] 08_bytetrack_feature_extraction"; fi
if stage_done 09_bytetrack_reference_evaluation; then echo "[skip] 09_bytetrack_reference_evaluation"; fi
if stage_done 18_root_cause_decision; then echo "[skip] 18_root_cause_decision"; fi
if stage_done 23_official_single_seed; then echo "[skip] 23_official_single_seed"; fi
if stage_done 24_single_seed_gate; then echo "[skip] 24_single_seed_gate"; fi
echo "PHASE4B_PIPELINE_FINISHED status=$(cat "$ROOT/runs/iclr27_phase4b/overall_status.txt")"
