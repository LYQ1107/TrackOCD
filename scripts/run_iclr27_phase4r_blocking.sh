#!/usr/bin/env bash
# Phase 4R / Q3 blocking orchestrator: Observation-Existence decision
# decoupling. Long jobs use setsid so they survive the launching shell.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase4r
DOCS=$ROOT/docs/iclr27_phase4r
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
PY=python3
Q2_CKPT=$ROOT/outputs/iclr27_phase4q/q2_long/checkpoint0007.pth
COMMON=(
  --config_file ./config/ovtr_lite_train_val.py
  --dataset_file lvis_generated_img_seqs --with_box_refine --two_stage
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13
  --num_workers 4 --batch_size 1
  --sample_mode random_interval --sample_interval 1
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3
  --track_query_iteration CIP --calculate_negative_samples --max_len 250
)
EVAL_COMMON=(
  --config_file ./config/ovtr_lite_train_val.py
  --dataset_file lvis_generated_img_seqs --batch_size 1
  --with_box_refine --two_stage --num_workers 4 --sampler_lengths 2
  --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19
  --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19
  --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45
  --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 --eval track
)

mkdir -p "$OUT"/{q2_alpha/a010,q2_alpha/a025,q3_pilot,q3_matched,q3_full,audits} "$DOCS"

launch() {
  local gpu=$1 out=$2; shift 2
  cd "$OVTR_DIR"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" "$@" \
    > "$out/cmd.log" 2>&1 &
  echo $! > "$out/run.pid"
}

wait_pid() {
  local pidfile=$1 name=$2
  while kill -0 "$(cat "$pidfile")" 2>/dev/null; do sleep 60; done
  echo "done waiting $name"
}

case "${1:-help}" in
  q2_alpha)
    launch 1 "$OUT/q2_alpha/a010" eval.py "${EVAL_COMMON[@]}" \
      --pretrain "$Q2_CKPT" --score_mode dscq --dscq_loss_coef 1.0 \
      --dscq_alpha 0.1 --output_dir "$OUT/q2_alpha/a010" \
      --result_path_track "$OUT/q2_alpha/a010/teta_results"
    launch 2 "$OUT/q2_alpha/a025" eval.py "${EVAL_COMMON[@]}" \
      --pretrain "$Q2_CKPT" --score_mode dscq --dscq_loss_coef 1.0 \
      --dscq_alpha 0.25 --output_dir "$OUT/q2_alpha/a025" \
      --result_path_track "$OUT/q2_alpha/a025/teta_results"
    wait_pid "$OUT/q2_alpha/a010/run.pid" a010
    wait_pid "$OUT/q2_alpha/a025/run.pid" a025
    for a in a010 a025; do
      PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
        --results-json "$OUT/q2_alpha/$a/teta_results/tao_track.json" \
        --out-prefix "$OUT/q2_alpha/$a/proposals"
    done ;;
  q3_pilot_train)
    launch 3 "$OUT/q3_pilot" main.py "${COMMON[@]}" \
      --epochs 2 --start_epoch 1 --max_train_iters 1000 --ckpt_interval 200 \
      --pretrain "$Q2_CKPT" --dscq_loss_coef 1.0 --dscq_alpha 0.5 \
      --dscq_detach_evidence 1 --dscq_state_dim 64 \
      --decision_decouple 1 --e_keep_thresh 0.5 --e_term_thresh 0.35 \
      --output_dir "$OUT/q3_pilot"
    wait_pid "$OUT/q3_pilot/run.pid" q3_pilot ;;
  q3_pilot_eval)
    launch 4 "$OUT/q3_pilot" eval.py "${EVAL_COMMON[@]}" \
      --pretrain "$OUT/q3_pilot/checkpoint0001.pth" --score_mode base \
      --dscq_loss_coef 1.0 --decision_decouple 1 \
      --e_keep_thresh 0.5 --e_term_thresh 0.35 \
      --output_dir "$OUT/q3_pilot" \
      --result_path_track "$OUT/q3_pilot/teta_results"
    wait_pid "$OUT/q3_pilot/run.pid" q3_pilot_eval
    PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
      --results-json "$OUT/q3_pilot/teta_results/tao_track.json" \
      --out-prefix "$OUT/q3_pilot/proposals" ;;
  q3_gate)
    cd "$OVTR_DIR"
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. "$OVTR_PY" \
      "$ROOT/src/iclr27_phase4r/q3_pilot_gate.py" \
      --gate-ckpt "$OUT/q3_pilot/checkpoint0001.pth" \
      --gate-out "$OUT/audits/q3_pilot_gate.json" --gate-iters 40 ;;
  help) sed -n '1,80p' "$0" ;;
  *) echo "usage: $0 {q2_alpha|q3_pilot_train|q3_pilot_eval|q3_gate|help}" ;;
esac
