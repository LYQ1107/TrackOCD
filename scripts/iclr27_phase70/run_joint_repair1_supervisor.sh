#!/usr/bin/env bash
set -u

# Repair supervisor for the interrupted Phase70 joint-d stage.  The first
# supervisor stopped while writing completion markers after /data1 filled.
# This namespace-local run keeps the old evidence and writes a fresh tag.
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
CFG="$ROOT/configs/iclr27_phase70/ovtr_phase70.py"
OUT="$ROOT/outputs/iclr27_phase70"
CK="$OUT/checkpoints"
COMP="$OUT/completion"
TAG=joint_d_repair1

mkdir -p "$CK" "$COMP" "$OUT/logs"
{
  echo "phase70_joint_repair1_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h /data1 /home/user
  echo "gpu_map=fold0:4 fold1:5 fold2:6 fold3:7"
  echo "schedule=joint_d repair; f0/f1/f3 from valid iter1000 (4000 steps), f2 from assign_c (5000 steps), batch=1 workers=1"
  echo "checkpoint_policy=interval-at-end-only; prior partial outputs retained"
} > "$OUT/joint_repair1_preflight.txt"

folds=(0 1 2 3)
gpus=(4 5 6 7)
pids=()
status=0

for i in "${!folds[@]}"; do
  f=${folds[$i]}; gpu=${gpus[$i]}
  out="$CK/${TAG}_f${f}"
  done="$COMP/${TAG}_f${f}.done"
  launched="$COMP/${TAG}_f${f}.launched"
  if [[ -f "$done" ]]; then
    echo "$TAG fold$f already done"
    continue
  fi
  if [[ -f "$launched" ]]; then
    echo "$TAG fold$f has launched marker without done; refusing duplicate" >&2
    status=2
    continue
  fi
  if [[ "$f" == 2 ]]; then
    prev="$CK/assign_c_f${f}/checkpoint.pth"
    steps=5000
    interval=5000
  else
    prev="$CK/joint_d_f${f}/checkpoint_iter001000.pth"
    steps=4000
    interval=4000
  fi
  if [[ ! -s "$prev" ]]; then
    echo "missing valid previous checkpoint: $prev" >&2
    status=3
    continue
  fi
  mkdir -p "$out"
  printf '{"fold":%d,"gpu":%d,"stage":"d","tag":"%s","steps":%d,"status":"launched","previous":"%s"}\n' \
    "$f" "$gpu" "$TAG" "$steps" "$prev" > "$launched.tmp"
  mv -f "$launched.tmp" "$launched"
  (
    cd "$OVTR"
    CUDA_VISIBLE_DEVICES="$gpu" PHASE70_FOLD="$f" PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" \
      "$ROOT/scripts/iclr27_phase70/train_semantic_fold.py" \
      --config_file "$CFG" --dataset_file lvis_generated_img_seqs --batch_size 1 --num_workers 1 \
      --with_box_refine --two_stage --track_query_iteration CIP --sampler_lengths 2 \
      --epochs 1 --max_train_iters "$steps" --ckpt_interval "$interval" --lr_drop 13 \
      --seed $((707000+f)) --pretrained "$prev" \
      --tco_loss_coef 1.0 --tco_alpha 0.5 --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1 \
      --dsct_stage d --dsct_no_unlabeled 1 --output_dir "$out"
  ) > "$out/train.log" 2>&1 &
  pids[$f]=$!
  echo "$TAG fold$f gpu$gpu pid=${pids[$f]} launched"
done

for f in "${folds[@]}"; do
  pid=${pids[$f]:-}
  [[ -n "$pid" ]] || continue
  if wait "$pid"; then
    out="$CK/${TAG}_f${f}"
    if [[ -s "$out/checkpoint.pth" ]] && grep -q "Training time" "$out/train.log"; then
      printf '{"fold":%d,"stage":"d","tag":"%s","status":"done"}\n' "$f" "$TAG" > "$COMP/${TAG}_f${f}.done.tmp"
      mv -f "$COMP/${TAG}_f${f}.done.tmp" "$COMP/${TAG}_f${f}.done"
      echo "$TAG fold$f complete"
    else
      status=1
      printf '{"fold":%d,"stage":"d","tag":"%s","status":"failed","reason":"missing_checkpoint_or_training_time"}\n' "$f" "$TAG" > "$COMP/${TAG}_f${f}.failed.tmp"
      mv -f "$COMP/${TAG}_f${f}.failed.tmp" "$COMP/${TAG}_f${f}.failed"
      echo "$TAG fold$f failed output contract" >&2
    fi
  else
    rc=$?
    status=1
    printf '{"fold":%d,"stage":"d","tag":"%s","status":"failed","exit_code":%d}\n' "$f" "$TAG" "$rc" > "$COMP/${TAG}_f${f}.failed.tmp"
    mv -f "$COMP/${TAG}_f${f}.failed.tmp" "$COMP/${TAG}_f${f}.failed"
    echo "$TAG fold$f failed rc=$rc" >&2
  fi
done

exit "$status"
