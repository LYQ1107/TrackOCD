#!/usr/bin/env bash
# Phase 6B ablation driver (GPU 7 and GPU 9): each ablation runs 5k Stage B
# + 5k Stage D from the Stage A checkpoint.
set -u

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase6b
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python

COMMON_ARGS=(
  --config_file ./config/ovtr_lite_dsct6b_train_val.py
  --dataset_file lvis_generated_img_seqs --with_box_refine --two_stage
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13
  --num_workers 4 --batch_size 1
  --sample_mode random_interval --sample_interval 1
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3
  --track_query_iteration CIP --calculate_negative_samples --max_len 250
  --tco_loss_coef 1.0 --tco_alpha 0.5
  --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1
  --dsct_known_coef 1.0 --dsct_disc_coef 1.0
  --dsct_struct_coef 1.0 --dsct_anchor_coef 0.1
  --dsct_novel_like_coef 2.0
  --epochs 1
)

is_alive() {
  local pidfile=$1
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
}

wait_pid() {
  local pidfile=$1
  if is_alive "$pidfile"; then
    echo "[wait] $(basename "$(dirname "$pidfile")") pid=$(cat "$pidfile")"
    while is_alive "$pidfile"; do
      sleep 60
    done
  fi
}

launch_train() {
  local gpu=$1 out=$2; shift 2
  mkdir -p "$out"
  cd "$OVTR_DIR"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" main.py \
    "${COMMON_ARGS[@]}" "$@" --output_dir "$out" > "$out/train.log" 2>&1 &
  echo $! > "$out/train.pid"
  echo "launched $out on GPU $gpu pid=$!"
}

run_ablation() {
  local name=$1 gpu=$2; shift 2
  local dir=$OUT/ablations/$name
  if [[ -f "$dir/.done" ]]; then
    echo "ablation $name already done"; return
  fi
  local stageb=$OUT/ablations/${name}_stageb
  if ! [[ -f "$stageb/.done" ]]; then
    if ! is_alive "$stageb/train.pid"; then
      launch_train "$gpu" "$stageb" --dsct_stage b --max_train_iters 5000 \
        --ckpt_interval 5000 \
        --resume "$OUT/training/stage_a/checkpoint.pth" "$@"
    fi
    wait_pid "$stageb/train.pid"
    grep -q "Training time" "$stageb/train.log" \
      || { echo "FAILED ablation stageb $name"; tail -30 "$stageb/train.log"; exit 1; }
    touch "$stageb/.done"
  fi
  if ! is_alive "$dir/train.pid"; then
    launch_train "$gpu" "$dir" --dsct_stage d --max_train_iters 5000 \
      --ckpt_interval 5000 \
      --resume "$stageb/checkpoint.pth" "$@"
  fi
  wait_pid "$dir/train.pid"
  grep -q "Training time" "$dir/train.log" \
    || { echo "FAILED ablation $name"; tail -30 "$dir/train.log"; exit 1; }
  touch "$dir/.done"
}

GPU=${ABLATION_GPU:-7}
LIST=${ABLATION_LIST:-"a2_no_p2s a4_no_struct"}
case "$LIST" in
  *a2_no_p2s*) run_ablation a2_no_p2s "$GPU" --dsct_disable_p2s 1 ;;
esac
case "$LIST" in
  *a4_no_struct*) run_ablation a4_no_struct "$GPU" --dsct_no_unlabeled_structure 1 ;;
esac
case "$LIST" in
  *a3_no_s2p*) run_ablation a3_no_s2p "$GPU" --dsct_disable_s2p 1 ;;
esac
case "$LIST" in
  *a5_knownconf*) run_ablation a5_knownconf "$GPU" --dsct_obj_ablation knownconf ;;
esac
echo "ABLATIONS_ALL_DONE"
