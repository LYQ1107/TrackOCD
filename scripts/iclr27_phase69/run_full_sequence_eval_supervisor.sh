#!/usr/bin/env bash
# Phase69 full-sequence evaluation of the four repaired OVTR checkpoints.
# This is a bounded, TRAIN/validation-only evaluation; no held/Q1/public
# labels are read. Each fold has an independent output and completion marker.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
OUTROOT="$ROOT/outputs/iclr27_phase69/metrics"
COMP="$ROOT/outputs/iclr27_phase69/completion"
CKPTROOT="$ROOT/outputs/iclr27_phase69/checkpoints"
TAG=repair1

mkdir -p "$OUTROOT" "$COMP"
{
  echo "phase69_full_sequence_eval_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1
  echo "gpu_map=fold0:4 fold1:5 fold2:6 fold3:7"
  echo "dataset=validation_ours_v1 (TRAIN/validation only)"
  echo "checkpoint_root=$CKPTROOT"
} > "$ROOT/outputs/iclr27_phase69/full_sequence_eval_preflight.txt"

declare -a pids=()
declare -a folds=(0 1 2 3)
declare -a gpus=(4 5 6 7)

for i in "${!folds[@]}"; do
  fold=${folds[$i]}
  gpu=${gpus[$i]}
  ckpt="$CKPTROOT/fold${fold}_${TAG}/checkpoint.pth"
  out="$OUTROOT/fold${fold}_eval"
  done_marker="$COMP/fold${fold}_eval.done"
  launched_marker="$COMP/fold${fold}_eval.launched"
  if [[ -f "$done_marker" ]]; then
    echo "fold${fold}: already done; skip"
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "fold${fold}: launched marker exists without done; refuse duplicate launch" >&2
    exit 2
  fi
  [[ -f "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 3; }
  mkdir -p "$out"
  tmp="${launched_marker}.tmp.$$"
  printf '{"fold":%d,"gpu":%d,"tag":"%s","dataset":"validation_ours_v1","status":"launched"}\n' "$fold" "$gpu" "$TAG" > "$tmp"
  mv -f "$tmp" "$launched_marker"
  (
    cd "$OVTR_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$PY" eval.py \
      --config_file ./config/ovtr_lite_train_val.py \
      --dataset_file lvis_generated_img_seqs --batch_size 1 \
      --with_box_refine --two_stage --pretrained "$ckpt" \
      --score_mode dsct --dsct_coef 1.0 --dsct_state_dim 128 \
      --dsct_alpha 0.1 --dsct_stage a --dsct_no_unlabeled 1 \
      --num_workers 1 --sampler_lengths 2 \
      --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
      --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
      --output_dir "$out" --eval track \
      --result_path_track "$out/teta_results"
  ) > "$out/eval.log" 2>&1 &
  pids[$fold]=$!
  echo "fold${fold} gpu${gpu} pid=${pids[$fold]} launched"
done

status=0
for fold in "${folds[@]}"; do
  pid=${pids[$fold]:-}
  [[ -n "$pid" ]] || continue
  if wait "$pid"; then
    if [[ ! -s "$OUTROOT/fold${fold}_eval/teta_results/tao_track.json" ]]; then
      status=1
      printf '{"fold":%d,"pid":%d,"tag":"%s","status":"failed","reason":"empty_tao_track"}\n' "$fold" "$pid" "$TAG" > "$COMP/fold${fold}_eval.failed"
      echo "fold${fold} completed with empty tao_track.json" >&2
    else
      done_marker="$COMP/fold${fold}_eval.done"
      tmp="${done_marker}.tmp.$$"
      printf '{"fold":%d,"pid":%d,"tag":"%s","status":"done","results":"%s"}\n' "$fold" "$pid" "$TAG" "$OUTROOT/fold${fold}_eval/teta_results/tao_track.json" > "$tmp"
      mv -f "$tmp" "$done_marker"
      echo "fold${fold} pid=${pid} completed"
    fi
  else
    rc=$?
    status=1
    printf '{"fold":%d,"pid":%d,"tag":"%s","status":"failed","exit_code":%d}\n' "$fold" "$pid" "$TAG" "$rc" > "$COMP/fold${fold}_eval.failed"
    echo "fold${fold} pid=${pid} failed rc=${rc}" >&2
  fi
done
exit "$status"
