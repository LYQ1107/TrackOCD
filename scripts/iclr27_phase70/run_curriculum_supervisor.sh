#!/usr/bin/env bash
set -euo pipefail

# Phase70 uses the existing OVTR DSCT/TCO implementation in a bounded
# semantic -> assign/create -> single joint curriculum.  Physical Phase69
# checkpoints are read-only inputs; every Phase70 output is namespace-local.
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
CFG="$ROOT/configs/iclr27_phase70/ovtr_phase70.py"
OUT="$ROOT/outputs/iclr27_phase70"
CK="$OUT/checkpoints"
COMP="$OUT/completion"
mkdir -p "$CK" "$COMP" "$OUT/logs"

{
  echo "phase70_curriculum_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1
  echo "gpu_map=fold0:4 fold1:5 fold2:6 fold3:7"
  echo "schedule=semantic_b:5000 assign_c:5000 joint_d:5000 steps, epochs=1, batch=1, workers=1"
  echo "physical_init=outputs/iclr27_phase69/checkpoints/fold*_repair1/checkpoint.pth"
} > "$OUT/curriculum_preflight.txt"

declare -a folds=(0 1 2 3)
declare -a gpus=(4 5 6 7)

run_stage() {
  local stage="$1"; local tag="$2"; local prev_kind="$3"
  declare -a pids=()
  for i in "${!folds[@]}"; do
    local f=${folds[$i]}; local gpu=${gpus[$i]}
    local out="$CK/${tag}_f${f}"
    local done="$COMP/${tag}_f${f}.done"
    local launched="$COMP/${tag}_f${f}.launched"
    local failed="$COMP/${tag}_f${f}.failed"
    if [[ -f "$done" ]]; then echo "$tag fold$f already done"; continue; fi
    if [[ -f "$launched" ]]; then echo "$tag fold$f has launched marker without done; refusing duplicate" >&2; return 2; fi
    local prev
    if [[ "$prev_kind" == "phase69" ]]; then
      prev="$ROOT/outputs/iclr27_phase69/checkpoints/fold${f}_repair1/checkpoint.pth"
    elif [[ "$prev_kind" == "semantic" ]]; then
      prev="$CK/semantic_b_f${f}/checkpoint.pth"
    else
      prev="$CK/assign_c_f${f}/checkpoint.pth"
    fi
    [[ -s "$prev" ]] || { echo "missing previous checkpoint $prev" >&2; return 3; }
    mkdir -p "$out"
    printf '{"fold":%d,"gpu":%d,"stage":"%s","steps":5000,"status":"launched"}\n' "$f" "$gpu" "$tag" > "$launched.tmp"
    mv -f "$launched.tmp" "$launched"
    (
      cd "$OVTR"
      CUDA_VISIBLE_DEVICES="$gpu" PHASE70_FOLD="$f" PYTHONPATH=. "$PY" "$ROOT/scripts/iclr27_phase70/train_semantic_fold.py" \
        --config_file "$CFG" --dataset_file lvis_generated_img_seqs --batch_size 1 --num_workers 1 \
        --with_box_refine --two_stage --track_query_iteration CIP --sampler_lengths 2 \
        --epochs 1 --max_train_iters 5000 --ckpt_interval 1000 --lr_drop 13 \
        --seed $((707000+f)) --pretrained "$prev" \
        --tco_loss_coef 1.0 --tco_alpha 0.5 --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1 \
        --dsct_stage "$stage" --dsct_no_unlabeled 1 --output_dir "$out"
    ) > "$out/train.log" 2>&1 &
    pids[$f]=$!
    echo "$tag fold$f gpu$gpu pid=${pids[$f]} launched"
  done
  local status=0
  for f in "${folds[@]}"; do
    local pid=${pids[$f]:-}; [[ -n "$pid" ]] || continue
    if wait "$pid"; then
      local out="$CK/${tag}_f${f}"
      if [[ -s "$out/checkpoint.pth" ]] && grep -q "Training time" "$out/train.log"; then
        printf '{"fold":%d,"stage":"%s","status":"done"}\n' "$f" "$tag" > "$COMP/${tag}_f${f}.done.tmp"
        mv -f "$COMP/${tag}_f${f}.done.tmp" "$COMP/${tag}_f${f}.done"
        echo "$tag fold$f complete"
      else
        status=1; printf '{"fold":%d,"stage":"%s","status":"failed","reason":"missing_checkpoint_or_training_time"}\n' "$f" "$tag" > "$COMP/${tag}_f${f}.failed.tmp"; mv -f "$COMP/${tag}_f${f}.failed.tmp" "$COMP/${tag}_f${f}.failed"
        echo "$tag fold$f failed output contract" >&2
      fi
    else
      local rc=$?; status=1
      printf '{"fold":%d,"stage":"%s","status":"failed","exit_code":%d}\n' "$f" "$tag" "$rc" > "$COMP/${tag}_f${f}.failed.tmp"; mv -f "$COMP/${tag}_f${f}.failed.tmp" "$COMP/${tag}_f${f}.failed"
      echo "$tag fold$f failed rc=$rc" >&2
    fi
  done
  return "$status"
}

run_stage b semantic_b phase69
run_stage c assign_c semantic
run_stage d joint_d assign
echo "PHASE70_CURRICULUM_DONE"
