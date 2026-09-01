#!/usr/bin/env bash
set -euo pipefail

# Bounded Phase69 supervisor: one OVTR adaptation worker per idle GPU 4--7.
# Workers are launched once only; a .launched marker is never silently
# retried.  This script intentionally keeps the historical Q0 schedule
# (15,000 updates/epoch, seven epochs) while adding the class-agnostic DSCT
# stage-A head through the Phase69 wrapper.
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR="$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
CFG="$ROOT/configs/iclr27_phase69/ovtr_phase69.py"
PRETRAIN="$ROOT/outputs/iclr27_phase4q/q0_long/checkpoint.pth"
OUTROOT="$ROOT/outputs/iclr27_phase69/checkpoints"
COMP="$ROOT/outputs/iclr27_phase69/completion"
mkdir -p "$OUTROOT" "$COMP"

{
  echo "phase69_formal_preflight=$(date -Iseconds)"
  echo "cwd=$ROOT"
  free -h
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "process_count=$(ps -e --no-headers | wc -l)"
  df -h "$ROOT"
  echo "gpu_map=fold0:4 fold1:5 fold2:6 fold3:7"
  echo "schedule=epochs:7 max_train_iters_per_epoch:15000 batch:1 workers:1"
  echo "pretrained=$PRETRAIN"
} > "$ROOT/outputs/iclr27_phase69/formal_preflight.txt"

declare -a pids=()
declare -a folds=(0 1 2 3)
declare -a gpus=(4 5 6 7)

for i in "${!folds[@]}"; do
  fold=${folds[$i]}; gpu=${gpus[$i]}
  out="$OUTROOT/fold${fold}"
  done_marker="$COMP/fold${fold}_formal.done"
  launched_marker="$COMP/fold${fold}_formal.launched"
  if [[ -f "$done_marker" ]]; then
    echo "fold${fold}: already done; skip"
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "fold${fold}: launched marker exists without done; refuse duplicate launch" >&2
    exit 2
  fi
  mkdir -p "$out"
  tmp="${launched_marker}.tmp.$$"
  printf '{"fold":%d,"gpu":%d,"epochs":7,"max_train_iters":15000,"status":"launched"}\n' "$fold" "$gpu" > "$tmp"
  mv -f "$tmp" "$launched_marker"
  (
    cd "$OVTR"
    CUDA_VISIBLE_DEVICES="$gpu" PHASE69_FOLD="$fold" PYTHONPATH=. "$PY" "$ROOT/scripts/iclr27_phase69/train_fold.py" \
      --config_file "$CFG" \
      --dataset_file lvis_generated_img_seqs --batch_size 1 --num_workers 1 \
      --with_box_refine --two_stage --track_query_iteration CIP --sampler_lengths 2 \
      --epochs 7 --max_train_iters 15000 --ckpt_interval 5000 --lr_drop 13 \
      --pretrained "$PRETRAIN" \
      --dsct_coef 1.0 --dsct_stage a --dsct_no_unlabeled 1 \
      --output_dir "$out"
  ) > "$out/train.log" 2>&1 &
  pids[$fold]=$!
  echo "fold${fold} gpu${gpu} pid=${pids[$fold]} launched"
done

status=0
for fold in "${folds[@]}"; do
  pid=${pids[$fold]:-}
  [[ -n "$pid" ]] || continue
  if wait "$pid"; then
    done_marker="$COMP/fold${fold}_formal.done"
    tmp="${done_marker}.tmp.$$"
    printf '{"fold":%d,"status":"done"}\n' "$fold" > "$tmp"
    mv -f "$tmp" "$done_marker"
    echo "fold${fold} pid=${pid} completed"
  else
    rc=$?
    status=1
    printf '{"fold":%d,"pid":%d,"status":"failed","exit_code":%d}\n' "$fold" "$pid" "$rc" > "$COMP/fold${fold}_formal.failed"
    echo "fold${fold} pid=${pid} failed rc=${rc}" >&2
  fi
done

exit "$status"
