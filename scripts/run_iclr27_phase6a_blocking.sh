#!/usr/bin/env bash
# Phase 6A blocking orchestrator: Joint End-to-End TrackOCD (JCDQ).
# Idempotent stages; long jobs are launched once with setsid and waited on
# with a single blocking wait. Markers: <dir>/.launched, <dir>/.done.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase6a
DOCS=$ROOT/docs/iclr27_phase6a
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
PY=python3
Q1_VIDEOS="[88,90,122,291,334,888,931,1159,1232,1276,1572,1865,2254,2347,2564,2675,2690,2759,2802,2888]"

COMMON_ARGS=(
  --config_file ./config/ovtr_lite_joint6a_train_val.py
  --dataset_file lvis_generated_img_seqs --with_box_refine --two_stage
  --lr 2e-4 --lr_backbone 2e-5 --lr_drop 13
  --num_workers 4 --batch_size 1
  --sample_mode random_interval --sample_interval 1
  --sampler_steps 4 7 14 --sampler_lengths 2 3 4 5
  --merger_dropout 0 --random_drop 0.1 --fp_ratio 0.3
  --track_query_iteration CIP --calculate_negative_samples --max_len 250
  --tco_loss_coef 1.0 --tco_alpha 0.5
  --joint_coef 1.0 --joint_alpha 0.1 --joint_state_dim 128
)

mkdir -p "$OUT"/{training,ablations,checkpoints,q1,strict_eval,physical_eval,semantic_eval,final} "$DOCS"

is_alive() {
  local pidfile=$1
  [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null
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

wait_train() {
  local out=$1
  if is_alive "$out/train.pid"; then
    while kill -0 "$(cat "$out/train.pid")" 2>/dev/null; do
      sleep 60
    done
  fi
  if ! grep -q "Training time" "$out/train.log"; then
    echo "FAILED: $out"; tail -60 "$out/train.log"; exit 1
  fi
  touch "$out/.done"
}

run_ablation() {
  local name=$1; shift
  local dir=$OUT/ablations/$name
  [[ -f "$dir/.done" ]] && { echo "ablation $name already done"; return; }
  if is_alive "$dir/train.pid"; then
    echo "ablation $name already running"
  else
    launch_train 9 "$dir" --epochs 1 --max_train_iters 5000 \
      --ckpt_interval 5000 \
      --resume "$ROOT/outputs/iclr27_phase4q/q1_long/checkpoint.pth" "$@"
    touch "$dir/.launched"
  fi
  wait_train "$dir"
}

eval_model() {
  local gpu=$1 ckpt=$2 out=$3; shift 3
  local eval_flags=(
    --config_file ./config/ovtr_lite_train_val.py
    --dataset_file lvis_generated_img_seqs --batch_size 1
    --with_box_refine --two_stage --pretrain "$ckpt"
    --score_mode joint --num_workers 4 --sampler_lengths 2
    --video_ids "$Q1_VIDEOS"
    --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19
    --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19
    --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45
    --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160
    --joint_coef 1.0 --joint_alpha 0.1 --joint_state_dim 128
    --joint_stats_path "$out/joint_stats.json"
  )
  mkdir -p "$out/teta_results"
  cd "$OVTR_DIR"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$OVTR_PY" eval.py \
    "${eval_flags[@]}" "$@" --output_dir "$out" --eval track \
    --result_path_track "$out/teta_results" > "$out/eval.log" 2>&1
  cd "$ROOT"
  PYTHONPATH="$ROOT" $PY "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
    --results-json "$out/teta_results/tao_track.json" \
    --out-prefix "$out/proposals"
}

extract_feats() {
  local csv=$1 out=$2
  PYTHONPATH="$ROOT" /home/lwr/anaconda3/envs/locatemot/bin/python \
    "$ROOT/src/iclr27_phase4s/features_q1.py" \
    --proposals "$csv" --out "$out" --device cuda:9 --batch 64 \
    > "$out/feats.log" 2>&1
}

strict_eval() {
  local csv=$1 out=$2 feats=$3
  local nrows=$(( $(wc -l < "$csv") - 1 ))
  if (( nrows <= 0 )); then
    mkdir -p "$out"
    cat > "$out/summary.json" <<EOF
{"strict":{},"legacy_first_frame":{},"legacy_last_frame":{},"n_rows":0,"n_records":0,"n_aligned_tracks":0}
EOF
  else
    PYTHONPATH="$ROOT" /home/lwr/anaconda3/envs/locatemot/bin/python \
      "$ROOT/src/iclr27_phase5a/evaluation/strict_causal_eval.py" \
      --proposals "$csv" \
      --feats "$feats" \
      --proto-dir outputs/iclr27_phase5a/pilot/episodes \
      --embed h --mode jointcsv --filter aligned --device cuda:9 \
      --out "$out"
  fi
}

physical_eval() {
  local csv=$1 out=$2
  PYTHONPATH="$ROOT" $PY "$ROOT/src/iclr27_phase6a/evaluation/physical_eval.py" \
    --csv "$csv" --out "$out"
}

stage_main_train() {
  local dir=$OUT/training/main
  [[ -f "$dir/.done" ]] && { echo "main training done"; return; }
  if is_alive "$dir/train.pid"; then
    echo "main training already running (pid $(cat "$dir/train.pid"))"
  else
    launch_train 0 "$dir" --epochs 1 --ckpt_interval 5000 \
      --resume "$ROOT/outputs/iclr27_phase4q/q1_long/checkpoint.pth"
  fi
  wait_train "$dir"
}

stage_ablations() {
  run_ablation a1_knownconf --joint_obj_ablation knownconf
  run_ablation a2_no_s2p --joint_disable_s2p 1
  run_ablation a3_no_p2s --joint_disable_p2s 1
  run_ablation a4_no_unlabeled --joint_disable_unlabeled 1
  run_ablation a5_no_dynamic_memory --joint_no_dynamic_memory 1
}

stage_eval() {
  local name=$1 ckpt=$2 extra=$3
  local out=$OUT/q1/$name
  local done_file=$out/.done
  [[ -f "$done_file" ]] && { echo "eval $name already done"; return; }
  mkdir -p "$out"
  # shellcheck disable=SC2086
  eval_model 9 "$ckpt" "$out" $extra
  extract_feats "$out/proposals_dev.csv" "$out"
  strict_eval "$out/proposals_dev.csv" "$OUT/strict_eval/${name}_joint" \
    "outputs/iclr27_phase6a/q1/$name/feats.npz"
  physical_eval "$out/proposals_dev.csv" "$OUT/physical_eval/${name}.json"
  PYTHONPATH="$ROOT" $PY "$ROOT/src/iclr27_phase6a/tests/causal_contract_tests.py" \
    --csv "$out/proposals_dev.csv" \
    --out "$OUT/strict_eval/${name}_causal_contract.json"
  PYTHONPATH="$ROOT" $PY "$ROOT/src/iclr27_phase6a/evaluation/objectness_audit.py" \
    --joint-stats "$out/joint_stats.json" \
    --out "$OUT/strict_eval/${name}_objectness_audit.json"
  touch "$done_file"
}

stage_all_eval() {
  stage_eval main "$OUT/training/main/checkpoint.pth" ""
  stage_eval a1_knownconf "$OUT/ablations/a1_knownconf/checkpoint.pth" "--joint_obj_ablation knownconf"
  stage_eval a2_no_s2p "$OUT/ablations/a2_no_s2p/checkpoint.pth" "--joint_disable_s2p 1"
  stage_eval a3_no_p2s "$OUT/ablations/a3_no_p2s/checkpoint.pth" "--joint_disable_p2s 1"
  stage_eval a4_no_unlabeled "$OUT/ablations/a4_no_unlabeled/checkpoint.pth" "--joint_disable_unlabeled 1"
  stage_eval a5_no_dynamic_memory "$OUT/ablations/a5_no_dynamic_memory/checkpoint.pth" "--joint_no_dynamic_memory 1"
}

case "${1:-all}" in
  train_main) stage_main_train ;;
  ablations) stage_ablations ;;
  eval) stage_all_eval ;;
  all)
    stage_main_train
    stage_ablations
    stage_all_eval
    ;;
  help) sed -n '1,80p' "$0" ;;
  *) echo "usage: $0 {train_main|ablations|eval|all|help}" ;;
esac
