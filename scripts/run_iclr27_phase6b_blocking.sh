#!/usr/bin/env bash
# Phase 6B blocking orchestrator: DSCT-TrackOCD.
#
# 00 preflight -> unit tests -> Stage A -> pilot (GPU9) -> pilot gate ->
# Stage B full -> Stage C -> Stage D -> Q1 frozen eval -> ablations (GPU7/9)
# -> final report.
#
# One blocking command; internal sleeps only. Idempotent via
# <unit>/.launched and <unit>/.done markers.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase6b
DOCS=$ROOT/docs/iclr27_phase6b
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
PY=python3
REP1=$ROOT/outputs/iclr27_phase6a/training/main_repair1/checkpoint.pth
Q1_VIDEOS="[88,90,122,291,334,888,931,1159,1232,1276,1572,1865,2254,2347,2564,2675,2690,2759,2802,2888]"

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

mkdir -p "$OUT"/{training,ablations,checkpoints,q1,strict_eval,physical_eval,semantic_eval,final,smoke,tests} "$DOCS"

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
  touch "$out/.launched"
  echo "launched $out on GPU $gpu pid=$!"
}

run_unit() {
  local name=$1; shift
  local dir=$OUT/training/$name
  local gpu=$1; shift
  if [[ -f "$dir/.done" ]]; then
    echo "unit $name already done"; return
  fi
  if is_alive "$dir/train.pid"; then
    echo "unit $name already running"
  else
    launch_train "$gpu" "$dir" "$@"
  fi
  wait_pid "$dir/train.pid"
  if ! grep -q "Training time" "$dir/train.log"; then
    echo "FAILED: $name"; tail -60 "$dir/train.log"; exit 1
  fi
  touch "$dir/.done"
}

eval_pilot() {
  local dir=$OUT/training/pilot_d
  if [[ -f "$dir/eval.done" ]]; then
    echo "pilot eval already done"; return
  fi
  mkdir -p "$OUT/q1/pilot_dsct"
  SKIP_STRICT=1 bash "$ROOT/scripts/eval_phase6b_one.sh" 9 \
    "$dir/checkpoint.pth" pilot_dsct --video_ids "$Q1_VIDEOS" \
    > "$OUT/q1/pilot_dsct/run.log" 2>&1
  touch "$dir/eval.done"
}

pilot_gate() {
  local dir=$OUT/training/pilot_d
  if [[ -f "$dir/gate.done" ]]; then
    echo "pilot gate already done"; return
  fi
  PYTHONPATH="$ROOT" $PY "$ROOT/src/iclr27_phase6b/evaluation/pilot_gate.py" \
    --joint-stats "$OUT/q1/pilot_dsct/joint_stats.json" \
    --train-logs "$OUT/training/pilot_b/train.log" \
                 "$OUT/training/pilot_c/train.log" \
                 "$OUT/training/pilot_d/train.log" \
    --objectness-audit "$OUT/strict_eval/pilot_dsct_objectness_audit.json" \
    --contract "$OUT/strict_eval/pilot_dsct_causal_contract.json" \
    --out "$OUT/training/pilot/gate.json"
  touch "$dir/gate.done"
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
    grep -q "Training time" "$stageb/train.log" || { echo "FAILED ablation stageb $name"; exit 1; }
    touch "$stageb/.done"
  fi
  if ! is_alive "$dir/train.pid"; then
    launch_train "$gpu" "$dir" --dsct_stage d --max_train_iters 5000 \
      --ckpt_interval 5000 \
      --resume "$stageb/checkpoint.pth" "$@"
  fi
  wait_pid "$dir/train.pid"
  grep -q "Training time" "$dir/train.log" || { echo "FAILED ablation $name"; exit 1; }
  touch "$dir/.done"
}

# ---------------- 00 preflight ----------------
free -h | head -2
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
df -h /data1 | tail -1

# ---------------- unit/contract tests ----------------
if [[ ! -f "$OUT/tests/.done" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$OVTR_DIR" "$OVTR_PY" \
    "$ROOT/src/iclr27_phase6b/tests/dsct_contract_tests.py" \
    "$OUT/tests/dsct_contract.json"
  touch "$OUT/tests/.done"
fi

# ---------------- Stage A: physical init (GPU 0) ----------------
run_unit stage_a 0 --dsct_stage a --max_train_iters 1000 --ckpt_interval 500 \
  --resume "$REP1" --dsct_init_joint "$REP1"

# ---------------- Pilot (GPU 9) ----------------
run_unit pilot_b 9 --dsct_stage b --max_train_iters 2000 --ckpt_interval 500 \
  --resume "$OUT/training/stage_a/checkpoint.pth"
run_unit pilot_c 9 --dsct_stage c --max_train_iters 5000 --ckpt_interval 1000 \
  --resume "$OUT/training/pilot_b/checkpoint.pth"
run_unit pilot_d 9 --dsct_stage d --max_train_iters 2000 --ckpt_interval 1000 \
  --resume "$OUT/training/pilot_c/checkpoint.pth"
eval_pilot
pilot_gate
echo "[gate] $(cat "$OUT/training/pilot/gate.json" 2>/dev/null || echo MISSING)"

# ---------------- Stage B: semantic representation (GPU 0) ----------------
run_unit stage_b 0 --dsct_stage b --max_train_iters 41421 --ckpt_interval 5000 \
  --resume "$OUT/training/stage_a/checkpoint.pth"

# ---------------- Ablations (GPU 7, 9; parallel with Stage C/D) ----------------
if [[ ! -f "$OUT/ablations/.launched" ]]; then
  touch "$OUT/ablations/.launched"
  nohup bash -c "
    $(declare -f run_unit launch_train wait_pid is_alive run_ablation)
    ROOT=$ROOT; OVTR_DIR=$OVTR_DIR; OUT=$OUT; OVTR_PY=$OVTR_PY
    COMMON_ARGS=(${COMMON_ARGS[*]})
    run_ablation a2_no_p2s 7 --dsct_disable_p2s 1
    run_ablation a4_no_struct 7 --dsct_no_unlabeled_structure 1
    run_ablation a3_no_s2p 9 --dsct_disable_s2p 1
    run_ablation a5_knownconf 9 --dsct_obj_ablation knownconf
  " >> "$OUT/ablations/driver.log" 2>&1 &
  echo $! > "$OUT/ablations/driver.pid"
  echo "ablations driver launched pid=$!"
fi

# ---------------- Stage C: assign/create (GPU 0) ----------------
run_unit stage_c 0 --dsct_stage c --max_train_iters 8000 --ckpt_interval 2000 \
  --resume "$OUT/training/stage_b/checkpoint.pth"

# ---------------- Stage D: joint causal finetune (GPU 0) ----------------
run_unit stage_d 0 --dsct_stage d --max_train_iters 8000 --ckpt_interval 2000 \
  --resume "$OUT/training/stage_c/checkpoint.pth"

# ---------------- Frozen Q1 full evaluation ----------------
if [[ ! -f "$OUT/q1/final_dsct/.done" ]]; then
  mkdir -p "$OUT/q1/final_dsct"
  FULL_VAL=1 SKIP_STRICT=0 bash "$ROOT/scripts/eval_phase6b_one.sh" 0 \
    "$OUT/training/stage_d/checkpoint.pth" final_dsct \
    > "$OUT/q1/final_dsct/run.log" 2>&1
  touch "$OUT/q1/final_dsct/.done"
fi

# ---------------- Wait for ablations ----------------
wait_pid "$OUT/ablations/driver.pid"

echo "PHASE6B_TRAINING_AND_EVAL_DONE"
