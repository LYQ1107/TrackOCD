#!/bin/bash
# Phase 4P: export frozen SimOWT/IDOL pre-association detections + replay
# packages for all 500 TAO train videos (8 chunks, 8 GPUs).
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
SIMOWT="$ROOT/third_party/SimOWT"
PY=/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python
EXPORT_DIR="$ROOT/outputs/iclr27_phase4p/train_export"
OUTROOT="$ROOT/runs/iclr27_phase4p/train_inference"
mkdir -p "$EXPORT_DIR" "$OUTROOT"

GPUS=(0 1 2 3 4 5 6 7)
PIDS=()
cd "$SIMOWT"
for k in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="${GPUS[$k]}" \
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  PYTHONPATH=. \
  SIMOWT_EXPORT_DIR="$EXPORT_DIR" \
  SIMOWT_OUTPUT_DIR="$OUTROOT/results_chunk$k" \
  "$PY" projects/IDOL/train_net.py \
    --config-file projects/IDOL/configs/r50_eval.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS "$ROOT/checkpoints/simowt_weight.pth" \
    DATASETS.TEST "('tao_train_agn_chunk$k',)" \
    INPUT.COCO_PRETRAIN True \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR "$OUTROOT/inference_chunk$k" \
    > "$OUTROOT/export_chunk$k.log" 2>&1 &
  PIDS+=($!)
done
FAIL=0
for p in "${PIDS[@]}"; do
  if ! wait "$p"; then FAIL=1; fi
done
echo "EXPORT_ALL_EXIT=$FAIL"
for k in 0 1 2 3 4 5 6 7; do
  n=$(cat "$EXPORT_DIR/pre_assoc_detections/"*".jsonl" 2>/dev/null | wc -l)
done
echo "TOTAL_PREASSOC_LINES=$(cat "$EXPORT_DIR"/pre_assoc_detections/*.jsonl | wc -l)"
exit "$FAIL"
