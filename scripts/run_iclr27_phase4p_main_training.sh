#!/usr/bin/env bash
# Phase 4P main training orchestrator (OVTR P0/P1/P2).
# P0 (1 epoch) is already running at the time this script was written;
# this file documents and re-runs each stage with one blocking command.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase4p/ovtr_main
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python

mkdir -p "$OUT"/{p0_official,p1_confirmation,p2_tco_pilot,p2_tco_epoch1}

common_train_args=(
  --config_file ./config/ovtr_lite_train_val.py
  --dataset_file lvis_generated_img_seqs
  --with_box_refine --two_stage
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13
  --num_workers 4 --batch_size 1
  --sample_mode random_interval --sample_interval 1
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3
  --track_query_iteration CIP --calculate_negative_samples --max_len 250
)

train_p0_epoch1() {
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. "$OVTR_PY" main.py \
    "${common_train_args[@]}" --epochs 1 \
    --pretrain ../model_zoo/ovtr_det_pretrain.pth \
    --ckpt_interval 10000 \
    --output_dir "$OUT/p0_official" \
    > "$OUT/p0_official/train.log" 2>&1
}

train_p2_pilot() {
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. "$OVTR_PY" main.py \
    "${common_train_args[@]}" --epochs 1 --max_train_iters 500 \
    --pretrain ../model_zoo/ovtr_det_pretrain.pth \
    --tco_loss_coef 1.0 --tco_alpha 0.5 --ckpt_interval 100 \
    --output_dir "$OUT/p2_tco_pilot" \
    > "$OUT/p2_tco_pilot/train.log" 2>&1
}

train_p2_epoch1() {
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. "$OVTR_PY" main.py \
    "${common_train_args[@]}" --epochs 1 \
    --pretrain ../model_zoo/ovtr_det_pretrain.pth \
    --tco_loss_coef 1.0 --tco_alpha 0.5 --ckpt_interval 10000 \
    --output_dir "$OUT/p2_tco_epoch1" \
    > "$OUT/p2_tco_epoch1/train.log" 2>&1
}

eval_ovtr() {
  local ckpt=$1 mode=$2 outdir=$3; shift 3
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. "$OVTR_PY" eval.py \
    --config_file ./config/ovtr_lite_train_val.py \
    --dataset_file lvis_generated_img_seqs --batch_size 1 \
    --with_box_refine --two_stage \
    --pretrain "$ckpt" --score_mode "$mode" "$@" \
    --num_workers 4 --sampler_lengths 2 \
    --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
    --output_dir "$outdir" \
    --eval track \
    --result_path_track "$outdir/teta_results" \
    > "$outdir/eval.log" 2>&1
}

eval_p0() {
  eval_ovtr "$OUT/p0_official/checkpoint.pth" base "$OUT/p0_official"
}

eval_p1() {
  eval_ovtr "$OUT/p0_official/checkpoint.pth" confirmation \
    "$OUT/p1_confirmation" --conf_w1 0.1 --conf_w2 0.1
}

eval_p2() {
  eval_ovtr "$OUT/p2_tco_epoch1/checkpoint.pth" tco "$OUT/p2_tco_epoch1" \
    --tco_loss_coef 1.0 --tco_alpha 0.5 \
    --tco_stats_path "$OUT/p2_tco_epoch1/tco_stats.json"
}

convert_dev_heldout() {
  local results=$1 outprefix=$2
  PYTHONPATH="$ROOT" python3 \
    "$ROOT/src/iclr27_phase4p/convert_teta_to_proposals.py" \
    --results-json "$results" --out-csv "${outprefix}_dev.csv" --mode dev
  PYTHONPATH="$ROOT" python3 \
    "$ROOT/src/iclr27_phase4p/convert_teta_to_proposals.py" \
    --results-json "$results" --out-csv "${outprefix}_heldout.csv" --mode heldout
}

case "${1:-help}" in
  p0) train_p0_epoch1 ;;
  p2_pilot) train_p2_pilot ;;
  p2_epoch) train_p2_epoch1 ;;
  eval_p0) eval_p0 ;;
  eval_p1) eval_p1 ;;
  eval_p2) eval_p2 ;;
  help) sed -n '1,80p' "$0" ;;
  *) echo "usage: $0 {p0|p2_pilot|p2_epoch|eval_p0|eval_p1|eval_p2|help}" ;;
esac
