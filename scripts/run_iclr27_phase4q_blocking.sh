#!/usr/bin/env bash
# Phase 4Q blocking orchestrator: Dual-State Causal Query (DSCQ).
# Each stage is idempotent; long training is launched with setsid so it
# survives the launching shell, and completion is waited on once with `wait`.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase4q
DOCS=$ROOT/docs/iclr27_phase4q
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
COMMON_ARGS=(
  --config_file ./config/ovtr_lite_train_val.py
  --dataset_file lvis_generated_img_seqs --with_box_refine --two_stage
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13
  --num_workers 4 --batch_size 1
  --sample_mode random_interval --sample_interval 1
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3
  --track_query_iteration CIP --calculate_negative_samples --max_len 250
)

mkdir -p "$OUT"/{q0_long,q1_long,q2_pilot,q2_long,p1plus,audits} "$DOCS"

launch_train() {
  local gpu=$1 out=$2; shift 2
  cd "$OVTR_DIR"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" main.py \
    "${COMMON_ARGS[@]}" "$@" --output_dir "$out" > "$out/train.log" 2>&1 &
  echo $! > "$out/train.pid"
  echo "launched $out on GPU $gpu pid=$!"
}

wait_train() {
  local out=$1
  while kill -0 "$(cat "$out/train.pid")" 2>/dev/null; do
    sleep 60
  done
  if ! grep -q "Training time" "$out/train.log"; then
    echo "FAILED: $out"; tail -40 "$out/train.log"; exit 1
  fi
}

eval_model() {
  local gpu=$1 ckpt=$2 mode=$3 out=$4; shift 4
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" eval.py \
    --config_file ./config/ovtr_lite_train_val.py \
    --dataset_file lvis_generated_img_seqs --batch_size 1 \
    --with_box_refine --two_stage --pretrain "$ckpt" \
    --score_mode "$mode" "$@" --num_workers 4 --sampler_lengths 2 \
    --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
    --output_dir "$out" --eval track \
    --result_path_track "$out/teta_results" > "$out/eval.log" 2>&1
}

convert_eval() {
  local results=$1 prefix=$2
  PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
    --results-json "$results" --out-prefix "$prefix"
}

case "${1:-help}" in
  q0_train)
    launch_train 1 "$OUT/q0_long" --epochs 8 --max_train_iters 15000 \
      --ckpt_interval 5000 \
      --resume "$ROOT/outputs/iclr27_phase4p/ovtr_main/p0_official/checkpoint.pth"
    wait_train "$OUT/q0_long" ;;
  q1_train)
    launch_train 2 "$OUT/q1_long" --epochs 8 --max_train_iters 15000 \
      --ckpt_interval 5000 --tco_loss_coef 1.0 --tco_alpha 0.5 \
      --resume "$ROOT/outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/checkpoint.pth"
    wait_train "$OUT/q1_long" ;;
  q2_pilot)
    launch_train 3 "$OUT/q2_pilot" --epochs 2 --start_epoch 1 \
      --max_train_iters 500 --ckpt_interval 100 \
      --pretrain "$ROOT/outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/checkpoint.pth" \
      --dscq_loss_coef 1.0 --dscq_alpha 0.5 --dscq_detach_evidence 1 --dscq_state_dim 64
    wait_train "$OUT/q2_pilot" ;;
  q2_train)
    launch_train 4 "$OUT/q2_long" --epochs 8 --start_epoch 1 \
      --max_train_iters 15000 --ckpt_interval 5000 \
      --pretrain "$ROOT/outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/checkpoint.pth" \
      --dscq_loss_coef 1.0 --dscq_alpha 0.5 --dscq_detach_evidence 1 --dscq_state_dim 64
    wait_train "$OUT/q2_long" ;;
  eval_q0)
    eval_model 1 "$OUT/q0_long/checkpoint.pth" base "$OUT/q0_long"
    convert_eval "$OUT/q0_long/teta_results/tao_track.json" "$OUT/q0_long/proposals" ;;
  eval_q1)
    eval_model 2 "$OUT/q1_long/checkpoint.pth" tco "$OUT/q1_long" \
      --tco_loss_coef 1.0 --tco_alpha 0.5
    convert_eval "$OUT/q1_long/teta_results/tao_track.json" "$OUT/q1_long/proposals" ;;
  eval_q2)
    eval_model 3 "$OUT/q2_long/checkpoint.pth" dscq "$OUT/q2_long" \
      --dscq_loss_coef 1.0 --dscq_alpha 0.5 \
      --dscq_stats_path "$OUT/q2_long/dscq_stats.json"
    convert_eval "$OUT/q2_long/teta_results/tao_track.json" "$OUT/q2_long/proposals" ;;
  p1plus)
    python3 "$ROOT/src/iclr27_phase4q/p1plus_confirmation.py" \
      --dev-csv "$ROOT/outputs/iclr27_phase4p/ovtr_main/p0_official/proposals_dev.csv" \
      --heldout-csv "$ROOT/outputs/iclr27_phase4p/ovtr_main/p0_official/proposals_heldout.csv" \
      --out-dir "$OUT/p1plus/on_p0"
    python3 "$ROOT/src/iclr27_phase4q/p1plus_confirmation.py" \
      --dev-csv "$ROOT/outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/proposals_dev.csv" \
      --heldout-csv "$ROOT/outputs/iclr27_phase4p/ovtr_main/p2_tco_epoch1/proposals_heldout.csv" \
      --out-dir "$OUT/p1plus/on_p2" ;;
  audit_q2)
    python3 "$ROOT/src/iclr27_phase4q/dscq_mechanism_audit.py" \
      --dscq-stats "$OUT/q2_long/dscq_stats.json" \
      --dev-csv "$OUT/q2_long/proposals_dev.csv" \
      --heldout-csv "$OUT/q2_long/proposals_heldout.csv" \
      --out-json "$OUT/audits/dscq_mechanism.json" ;;
  help) sed -n '1,80p' "$0" ;;
  *) echo "usage: $0 {q0_train|q1_train|q2_pilot|q2_train|eval_q0|eval_q1|eval_q2|p1plus|audit_q2|help}" ;;
esac
