#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
OUTROOT="$ROOT/outputs/iclr27_phase70/validation/joint_d_repair1"
COMP="$ROOT/outputs/iclr27_phase70/completion"
TAG=joint_d_repair1_validation
mkdir -p "$OUTROOT" "$COMP"
{
  echo "phase70_validation_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1 /home/user
  echo "gpu_map=fold0:4 fold1:5 fold2:6 fold3:7"
  echo "dataset=validation_ours_v1; labels_for_model=false"
} > "$ROOT/outputs/iclr27_phase70/validation_preflight.txt"

folds=(0 1 2 3)
gpus=(4 5 6 7)
pids=()
status=0
for i in "${!folds[@]}"; do
  fold=${folds[$i]}; gpu=${gpus[$i]}
  ckpt="$ROOT/outputs/iclr27_phase70/checkpoints/joint_d_repair1_f${fold}/checkpoint.pth"
  out="$OUTROOT/fold${fold}_eval"
  done_marker="$COMP/${TAG}_f${fold}.done"
  launched_marker="$COMP/${TAG}_f${fold}.launched"
  if [[ -f "$done_marker" ]]; then continue; fi
  if [[ -f "$launched_marker" ]]; then echo "refusing duplicate validation launch for fold${fold}" >&2; exit 2; fi
  [[ -s "$ckpt" ]] || { echo "missing checkpoint $ckpt" >&2; exit 3; }
  mkdir -p "$out"
  printf '{"fold":%d,"gpu":%d,"tag":"%s","status":"launched","checkpoint":"%s"}\n' "$fold" "$gpu" "$TAG" "$ckpt" > "$launched_marker.tmp"
  mv -f "$launched_marker.tmp" "$launched_marker"
  (
    cd "$OVTR_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" eval.py \
      --config_file ./config/ovtr_lite_train_val.py --dataset_file lvis_generated_img_seqs --batch_size 1 \
      --with_box_refine --two_stage --pretrained "$ckpt" --score_mode dsct --dsct_coef 1.0 --dsct_state_dim 128 \
      --dsct_alpha 0.1 --dsct_stage a --dsct_no_unlabeled 1 --num_workers 1 --sampler_lengths 2 \
      --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
      --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
      --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
      --output_dir "$out" --eval track --result_path_track "$out/teta_results"
  ) > "$out/eval.log" 2>&1 &
  pids[$fold]=$!
  echo "$TAG fold${fold} gpu${gpu} pid=${pids[$fold]} launched"
done

for fold in "${folds[@]}"; do
  pid=${pids[$fold]:-}
  [[ -n "$pid" ]] || continue
  if wait "$pid" && [[ -s "$OUTROOT/fold${fold}_eval/teta_results/tao_track.json" ]]; then
    printf '{"fold":%d,"pid":%d,"tag":"%s","status":"done","results":"%s"}\n' "$fold" "$pid" "$TAG" "$OUTROOT/fold${fold}_eval/teta_results/tao_track.json" > "$COMP/${TAG}_f${fold}.done.tmp"
    mv -f "$COMP/${TAG}_f${fold}.done.tmp" "$COMP/${TAG}_f${fold}.done"
  else
    rc=$?
    status=1
    printf '{"fold":%d,"pid":%d,"tag":"%s","status":"failed","exit_code":%d}\n' "$fold" "$pid" "$TAG" "$rc" > "$COMP/${TAG}_f${fold}.failed.tmp"
    mv -f "$COMP/${TAG}_f${fold}.failed.tmp" "$COMP/${TAG}_f${fold}.failed"
  fi
done
exit "$status"
