#!/usr/bin/env bash
# ICLR27 Phase 3A blocking pipeline (resumable).
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
SIMOWT_PY="/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python"
MK="$ROOT/runs/iclr27_phase3a/markers"
mkdir -p "$MK" "$ROOT/runs/iclr27_phase3a" "$ROOT/outputs/iclr27_phase3a/audit" \
  "$ROOT/outputs/iclr27_phase3a/smoke" "$ROOT/outputs/iclr27_phase3a/tests"
stage_ok() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_ok "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
skipped() { touch "$MK/$1.skipped"; echo "[skipped] $1"; }

s00() { if ! begin 00_preflight; then return 0; fi
  [[ -f "$ROOT/outputs/simowt/val_predictions.json" ]]; finish 00_preflight; }
s01() { if ! begin 01_read_agents; then return 0; fi
  [[ -f "$ROOT/AGENTS.md" ]]; finish 01_read_agents; }
s02() { if ! begin 02_read_prior_audits; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase2_5/PHASE2_5_FINAL_REPORT.md" ]]; finish 02_read_prior_audits; }
s03() { if ! begin 03_hash_inputs; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase3a/audit/input_hashes.json" ]]; finish 03_hash_inputs; }
s04() { if ! begin 04_association_dependency_audit; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase3a/ASSOCIATION_DEPENDENCY_AUDIT.md" ]]; finish 04_association_dependency_audit; }
s05() { if ! begin 05_select_20_videos; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/select_20.py"; finish 05_select_20_videos; }
s06() { if ! begin 06_implement_export_schema; then return 0; fi
  [[ -f "$ROOT/outputs/iclr27_phase3a/smoke/export_schema.json" ]]; finish 06_implement_export_schema; }
s07() { if ! begin 07_patch_writer; then return 0; fi
  [[ -f "$ROOT/patches/iclr27_phase3a/simowt_pre_assoc_export.patch" ]]; finish 07_patch_writer; }
s08() { if ! begin 08_writer_insertion_tests; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase3a/WRITER_INSERTION_AUDIT.md" ]]; finish 08_writer_insertion_tests; }
s09() { if ! begin 09_run_instrumented_online_20; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/build_coco_20_json.py"
  mkdir -p "$ROOT/outputs/iclr27_phase3a/trajectories/instrumented_online_20" \
    "$ROOT/outputs/iclr27_phase3a/smoke"
  cd "$ROOT/third_party/SimOWT"
  CUDA_VISIBLE_DEVICES="${PHASE3A_GPU:-0}" \
  SIMOWT_COCO_20_JSON="$ROOT/outputs/iclr27_phase3a/smoke/tao_subset/validation_20_coco.json" \
  SIMOWT_EXPORT_DIR="$ROOT/outputs/iclr27_phase3a/smoke" \
  SIMOWT_OUTPUT_DIR="$ROOT/outputs/iclr27_phase3a/trajectories/instrumented_online_20/" \
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  PYTHONPATH=. \
  "$SIMOWT_PY" projects/IDOL/train_net.py \
    --config-file projects/IDOL/configs/r50_eval.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS "$ROOT/checkpoints/simowt_weight.pth" \
    DATALOADER.NUM_WORKERS 4 \
    DATASETS.TEST "('coco_2017_val_agn_20',)" \
    INPUT.COCO_PRETRAIN True \
    OUTPUT_DIR "$ROOT/runs/iclr27_phase3a/inference_20_coco"
  cd "$ROOT"
  finish 09_run_instrumented_online_20; }
s10() { if ! begin 10_extract_original_outputs_20; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/extract_original_20.py"
  finish 10_extract_original_outputs_20; }
s11() { if ! begin 11_compare_original_vs_instrumented; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/analyze_phase3a.py"
  finish 11_compare_original_vs_instrumented; }
s12() { if ! begin 12_build_offline_replay; then return 0; fi
  [[ -f "$ROOT/src/iclr27_phase3a/run_offline_replay.py" ]]
  finish 12_build_offline_replay; }
s13() { if ! begin 13_run_offline_replay_20; then return 0; fi
  CUDA_VISIBLE_DEVICES="${PHASE3A_REPLAY_GPU:-${PHASE3A_GPU:-1}}" \
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  "$SIMOWT_PY" "$ROOT/src/iclr27_phase3a/run_offline_replay.py"
  finish 13_run_offline_replay_20; }
s14() { if ! begin 14_compare_instrumented_vs_replay; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/build_tracker_json.py" \
    --input-dir "$ROOT/outputs/iclr27_phase3a/trajectories/offline_replay_20" \
    --output-json "$ROOT/outputs/iclr27_phase3a/trackeval/trackers/offline_replay/data/pred.json"
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  "$SIMOWT_PY" "$ROOT/src/iclr27_phase3a/run_trackeval.py"
  "$PY" "$ROOT/src/iclr27_phase3a/analyze_phase3a.py"
  finish 14_compare_instrumented_vs_replay; }
s15() { if ! begin 15_pre_post_association_analysis; then return 0; fi
  "$PY" "$ROOT/src/iclr27_phase3a/analyze_phase3a.py"
  finish 15_pre_post_association_analysis; }
s16() { if ! begin 16_unit_tests; then return 0; fi
  "$PY" "$ROOT/tests/iclr27_phase3a/test_phase3a.py"
  finish 16_unit_tests; }
s17() { if ! begin 17_roundtrip_gate; then return 0; fi
  gate="$ROOT/outputs/iclr27_phase3a/fidelity/roundtrip_gate.json"
  if python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall"])' "$gate" | grep -q True; then
    printf 'ROUNDTRIP_VALIDATED_READY_FOR_FULL_EXPORT\n' > "$ROOT/runs/iclr27_phase3a/status.txt"
  else
    printf 'ROUNDTRIP_FIDELITY_FAILED\n' > "$ROOT/runs/iclr27_phase3a/status.txt"
  fi
  finish 17_roundtrip_gate; }
s18() { if ! begin 18_final_report; then return 0; fi
  [[ -f "$ROOT/docs/iclr27_phase3a/PHASE3A_FINAL_REPORT.md" ]]; finish 18_final_report; }
s19() { if ! begin 19_mark_full_export_skipped; then return 0; fi
  skipped 19_mark_full_export_skipped; }
s20() { if ! begin 20_mark_bytetrack_skipped; then return 0; fi
  skipped 20_mark_bytetrack_skipped; }

s00; s01; s02; s03; s04; s05; s06; s07; s08; s09; s10; s11; s12; s13; s14
s15; s16; s17; s18; s19; s20
echo "PIPELINE_FINISHED status=$(cat "$ROOT/runs/iclr27_phase3a/status.txt" 2>/dev/null || echo PENDING)"
