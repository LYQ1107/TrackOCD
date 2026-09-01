#!/usr/bin/env bash
# ICLR27 Phase 3B-4A blocking supervisor (Workstream A + B + C).
set -u
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
SIMOWT_PY="/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python"
MK="$ROOT/runs/iclr27_phase3b/markers"
mkdir -p "$MK"

stage_done() { [[ -f "$MK/$1.done" ]]; }
begin() { if stage_done "$1"; then echo "[skip] $1"; return 1; fi; echo "[run ] $1"; touch "$MK/$1.launched"; return 0; }
finish() { echo "$(date -u +%FT%TZ) $1" > "$MK/$1.done.tmp"; mv "$MK/$1.done.tmp" "$MK/$1.done"; echo "[done] $1"; }
failed() { echo "$2" > "$MK/$1.failed"; echo "[failed] $1: $2"; }

# Workstream A full export (launched manually in current session; script
# supports rerun from scratch).
if ! stage_done 01_full_export; then
  touch "$MK/01_full_export.launched"
  mkdir -p "$ROOT/outputs/iclr27_phase3b/full_export/instrumented_online"
  cd "$ROOT/third_party/SimOWT"
  CUDA_VISIBLE_DEVICES="${PHASE3B_GPU:-4}" \
  SIMOWT_EXPORT_DIR="$ROOT/outputs/iclr27_phase3b/full_export" \
  SIMOWT_OUTPUT_DIR="$ROOT/outputs/iclr27_phase3b/full_export/instrumented_online/" \
  SIMOWT_REPLAY_LITE=1 \
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  PYTHONPATH=. \
  "$SIMOWT_PY" projects/IDOL/train_net.py \
    --config-file projects/IDOL/configs/r50_eval.yaml --num-gpus 1 --eval-only \
    MODEL.WEIGHTS "$ROOT/checkpoints/simowt_weight.pth" \
    DATALOADER.NUM_WORKERS 4 \
    DATASETS.TEST "('coco_2017_val_agn',)" \
    INPUT.COCO_PRETRAIN True \
    OUTPUT_DIR "$ROOT/runs/iclr27_phase3b/full_export_inference" \
    > "$ROOT/runs/iclr27_phase3b/full_export.log" 2>&1
  if [[ $? -ne 0 ]]; then failed 01_full_export "simowt full export failed"; else finish 01_full_export; fi
fi

# ORBIT reference / baselines (already completed in this session).
if begin 02_orbit_reference; then
  (cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" src/orbit/reference.py) \
    && finish 02_orbit_reference || failed 02_orbit_reference "reference reproduction failed"
fi
if begin 03_baseline_ladder; then
  (cd "$ROOT" && PYTHONPATH="$ROOT" "$PY" src/orbit/baselines.py) \
    && finish 03_baseline_ladder || failed 03_baseline_ladder "baseline ladder failed"
fi
if begin 04_orbit_train; then
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="${ORBIT_GPU:-5}" PYTHONPATH="$ROOT" "$PY" \
    src/orbit/train.py --variant D1 --epochs 30 --episodes_per_epoch 10 \
    --num_known 20 --support_per_class 4 --query_per_class 4 \
    --bottleneck 128 --lambda_geo 0.3 --seed 1027) \
    && finish 04_orbit_train || failed 04_orbit_train "orbit training failed"
fi
if begin 05_orbit_final; then
  (cd "$ROOT" && CUDA_VISIBLE_DEVICES="${ORBIT_GPU:-5}" PYTHONPATH="$ROOT" "$PY" src/orbit/run_final.py) \
    && finish 05_orbit_final || failed 05_orbit_final "orbit final eval failed"
fi

echo "PIPELINE_FINISHED"
