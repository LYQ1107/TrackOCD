#!/usr/bin/env bash
set -euo pipefail

PROJ=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$PROJ"
mkdir -p runs/arch1_main

DISCOVERY_PY=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python
SIMOWT_PY=/home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python
export PYTHONPATH="$PROJ"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-8}"

run_stage() {
  local name="$1"
  shift
  local launched="runs/arch1_main/${name}.launched"
  local done_marker="runs/arch1_main/${name}.done"
  if [[ -f "$done_marker" ]]; then
    echo "[skip] $name already done"
    return 0
  fi
  if [[ -f "$launched" ]]; then
    local pid
    pid="$(cat "$launched" 2>/dev/null || echo 0)"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[wait] $name already running (pid $pid)"
      while kill -0 "$pid" 2>/dev/null; do sleep 10; done
    else
      echo "[resume] stale marker removed for $name"
      rm -f "$launched"
    fi
  fi
  echo "=== STAGE $name ==="
  touch "$launched"
  echo $$ > "$launched"
  if ! "$@"; then
    echo "STAGE $name FAILED"
    rm -f "$launched"
    exit 1
  fi
  touch "$done_marker"
  rm -f "$launched"
  echo "=== STAGE $name DONE ==="
}

memory_ok() {
  local avail_kb
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  local total_kb
  total_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  local ratio
  ratio=$(python3 -c "print(${avail_kb}/${total_kb})")
  if python3 -c "import sys; sys.exit(0 if $ratio >= 0.25 else 1)"; then
    echo "free RAM ratio: $ratio (>=0.25 OK)"
    return 0
  fi
  echo "free RAM ratio: $ratio (<0.25, aborting before long job)" >&2
  return 1
}

stage_preflight() {
  memory_ok
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > runs/arch1_main/gpu_preflight.csv
  cat runs/arch1_main/gpu_preflight.csv
}

stage_dataset() {
  "$DISCOVERY_PY" src/data/build_protocol.py --scan-frames
}

stage_extract_gt() {
  "$DISCOVERY_PY" src/features/extract.py --encoder dinov2 --split gt_val --mode mean
  "$DISCOVERY_PY" src/features/extract.py --encoder dinov2 --split gt_val --mode single
  "$DISCOVERY_PY" src/features/extract.py --encoder dinov2 --split train_known --mode mean
}

stage_gt_offline() {
  "$DISCOVERY_PY" src/ocd/offline_kmeans.py --encoder dinov2 --mode single --subset full
  "$DISCOVERY_PY" src/ocd/offline_kmeans.py --encoder dinov2 --mode mean --subset full
  "$DISCOVERY_PY" src/ocd/offline_kmeans.py --encoder dinov2 --mode mean --subset repeated
  "$DISCOVERY_PY" src/ocd/offline_kmeans.py --encoder dinov2 --mode mean --subset balanced
}

stage_train_phe() {
  for seed in 1027 1028 1029; do
    "$DISCOVERY_PY" src/ocd/phe_track/train_phe_track.py --encoder dinov2 --seed "$seed"
  done
}

stage_gt_online_phe() {
  for seed in 1027 1028 1029; do
    "$DISCOVERY_PY" src/ocd/phe_track/eval_phe_track.py --encoder dinov2 --seed "$seed" --subset full
  done
}

stage_simowt_inference() {
  cd "$PROJ/third_party/SimOWT"
  rm -rf "$PROJ/runs/simowt_inference"
  mkdir -p "$PROJ/runs/simowt_inference"
  SIMOWT_OUTPUT_DIR="$PROJ/runs/simowt_inference" \
  LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
  PYTHONPATH=. \
  "$SIMOWT_PY" projects/IDOL/train_net.py \
    --config-file projects/IDOL/configs/r50_eval.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS "$PROJ/checkpoints/simowt_weight.pth" \
    DATALOADER.NUM_WORKERS 4 \
    OUTPUT_DIR "$PROJ/runs/simowt_inference"
  cd "$PROJ"
}

stage_trackeval() {
  "$DISCOVERY_PY" "$PROJ/scripts/merge_simowt_output.py" \
    --input-dir "$PROJ/runs" \
    --output-json outputs/simowt/val_predictions.json \
    --stream-jsonl data/tao_ow_ocd_v1/public/pred_track_stream.jsonl
  mkdir -p third_party/TrackEval/data/trackers/tao/tao_training/simowt/data
  cp outputs/simowt/val_predictions.json third_party/TrackEval/data/trackers/tao/tao_training/simowt/data/pred.json
  for subset in known unknown; do
    cd third_party/TrackEval
    LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
    "$SIMOWT_PY" scripts/run_tao_ow.py \
      --USE_PARALLEL False --METRICS HOTA \
      --TRACKERS_TO_EVAL simowt --SUBSET "$subset" \
      --GT_FOLDER data/gt/tao/tao_training \
      --TRACKERS_FOLDER data/trackers/tao/tao_training
    cd "$PROJ"
  done
}

stage_pred_features() {
  "$DISCOVERY_PY" src/evaluation/track_matching.py
  "$DISCOVERY_PY" src/features/extract.py --encoder dinov2 --split pred_val --mode mean --sampling score \
    --stream pred_track_stream_matched_iou0.5.jsonl
}

stage_pred_phe() {
  for seed in 1027 1028 1029; do
    "$DISCOVERY_PY" src/ocd/phe_track/eval_pred_phe.py --encoder dinov2 --seed "$seed" --iou 0.5
  done
}

stage_final_eval() {
  "$DISCOVERY_PY" src/evaluation/summarize.py
}

STAGE="${1:-all}"
case "$STAGE" in
  all)
    run_stage 00_preflight stage_preflight
    run_stage 02_dataset_scan stage_dataset
    run_stage 04_extract_gt_features stage_extract_gt
    run_stage 05_gt_offline_baselines stage_gt_offline
    run_stage 06_train_phe_track stage_train_phe
    run_stage 07_gt_online_phe stage_gt_online_phe
    run_stage 08_simowt_inference stage_simowt_inference
    run_stage 09_trackeval stage_trackeval
    run_stage 10_pred_track_features stage_pred_features
    run_stage 11_pred_track_phe stage_pred_phe
    run_stage 12_final_evaluation stage_final_eval
    ;;
  *)
    "stage_$STAGE"
    ;;
esac
