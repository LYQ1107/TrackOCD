#!/usr/bin/env bash
# Phase71 full-sequence validation with one evaluator process at a time.
# This is TRAIN/validation-only and uses the frozen registered evaluator.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
RUN_TAG=${1:-formal1_tco_serial}
CKPT_TAG=${2:-formal1}
OUTROOT="$ROOT/outputs/iclr27_phase71/validation/$RUN_TAG"
COMP="$ROOT/outputs/iclr27_phase71/completion"
CKPTROOT="$ROOT/outputs/iclr27_phase71/runs/$CKPT_TAG"
GPU=${PHASE71_EVAL_GPU:-4}

mkdir -p "$OUTROOT" "$COMP"
{
  echo "phase71_eval_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1 /data2
  echo "gpu_map=serial:${GPU}"
  echo "dataset=validation_ours_v1 (TRAIN/validation only)"
  echo "checkpoint_root=$CKPTROOT"
  echo "score_mode=tco tco_loss_coef=1.0 tco_alpha=0.5"
  echo "concurrency=1"
} > "$OUTROOT/preflight.txt"

total_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
floor_kib=$(( total_kib * 25 / 100 ))
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < floor_kib )); then
  echo "available RAM ${available_kib} KiB below 25% floor ${floor_kib} KiB" >&2
  exit 4
fi

status=0
for fold in 0 1 2 3; do
  ckpt="$CKPTROOT/fold_$fold/checkpoint.pth"
  out="$OUTROOT/fold_$fold"
  done_marker="$COMP/${RUN_TAG}_f${fold}.done"
  launched_marker="$COMP/${RUN_TAG}_f${fold}.launched"
  failed_marker="$COMP/${RUN_TAG}_f${fold}.failed"
  if [[ -f "$done_marker" ]]; then
    echo "fold${fold}: already done; skip"
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "fold${fold}: launched marker exists without done; refusing duplicate launch" >&2
    exit 2
  fi
  [[ -s "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 3; }
  available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_kib < floor_kib )); then
    printf '{"fold":%d,"tag":"%s","status":"failed","reason":"resource_memory_floor_before_fold","available_kib":%d,"floor_kib":%d}\n' "$fold" "$RUN_TAG" "$available_kib" "$floor_kib" > "${failed_marker}.tmp.$$"
    mv -f "${failed_marker}.tmp.$$" "$failed_marker"
    exit 5
  fi
  mkdir -p "$out"
  tmp="${launched_marker}.tmp.$$"
  printf '{"fold":%d,"gpu":%d,"tag":"%s","dataset":"validation_ours_v1","score_mode":"tco","status":"launched","concurrency":1}\n' "$fold" "$GPU" "$RUN_TAG" > "$tmp"
  mv -f "$tmp" "$launched_marker"
  echo "fold${fold} gpu${GPU} started serially"
  set +e
  (
    cd "$OVTR_DIR"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" eval.py \
      --config_file ./config/ovtr_lite_train_val.py \
      --dataset_file lvis_generated_img_seqs --batch_size 1 \
      --with_box_refine --two_stage --pretrained "$ckpt" \
      --tco_loss_coef 1.0 --tco_alpha 0.5 --score_mode tco \
      --num_workers 1 --sampler_lengths 2 \
      --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
      --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
      --output_dir "$out" --eval track \
      --result_path_track "$out/teta_results"
  ) > "$out/eval.log" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '{"fold":%d,"gpu":%d,"tag":"%s","status":"failed","exit_code":%d,"reason":"evaluator_exit"}\n' "$fold" "$GPU" "$RUN_TAG" "$rc" > "${failed_marker}.tmp.$$"
    mv -f "${failed_marker}.tmp.$$" "$failed_marker"
    status=1
    echo "fold${fold} failed rc=${rc}" >&2
    break
  fi
  if [[ ! -s "$out/teta_results/tao_track.json" ]]; then
    printf '{"fold":%d,"gpu":%d,"tag":"%s","status":"failed","reason":"empty_tao_track"}\n' "$fold" "$GPU" "$RUN_TAG" > "${failed_marker}.tmp.$$"
    mv -f "${failed_marker}.tmp.$$" "$failed_marker"
    status=1
    echo "fold${fold} completed with empty tao_track.json" >&2
    break
  fi
  done_tmp="${done_marker}.tmp.$$"
  printf '{"fold":%d,"gpu":%d,"tag":"%s","status":"done","results":"%s"}\n' "$fold" "$GPU" "$RUN_TAG" "$out/teta_results/tao_track.json" > "$done_tmp"
  mv -f "$done_tmp" "$done_marker"
  echo "fold${fold} completed serially"
done
exit "$status"
