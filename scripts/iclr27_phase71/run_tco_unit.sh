#!/usr/bin/env bash
set -euo pipefail

# One bounded unit.  The supervisor owns marker creation; this script only
# runs the immutable OVTR trainer and exits with its status.
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
FOLD="${PHASE71_FOLD:?PHASE71_FOLD required}"
TAG="${PHASE71_TAG:?PHASE71_TAG required}"
MODE="${PHASE71_MODE:?PHASE71_MODE required}"
GPU="${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES required}"
OUT="$ROOT/outputs/iclr27_phase71/runs/${TAG}/fold_${FOLD}"
mkdir -p "$OUT"

case "$MODE" in
  smoke) EPOCHS=1; MAX_ITERS=20; CKPT=20; BATCH=1; WORKERS=0; SAMPLER=(2); STEPS=() ;;
  targeted) EPOCHS=1; MAX_ITERS=100; CKPT=100; BATCH=1; WORKERS=0; SAMPLER=(2); STEPS=() ;;
  # Q0-equivalent update budget: seven epoch passes capped at the historical
  # 15k iterations/epoch, with the original 2→3→4→5 frame curriculum.
  formal) EPOCHS=7; MAX_ITERS=15000; CKPT=5000; BATCH=1; WORKERS=2; SAMPLER=(2 3 4 5); STEPS=(4 7 14) ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ "${#STEPS[@]}" -gt 0 ]]; then
  EXTRA_ARGS+=(--sampler_steps "${STEPS[@]}")
fi

cd "$ROOT/third_party/research_refs_phase4n/OVTR/ovtr"
exec /home/lwr/anaconda3/envs/ovtr/bin/python "$ROOT/scripts/iclr27_phase71/train_tco_fold.py" \
  --config_file "$ROOT/configs/iclr27_phase71/phase71_q0_adapter.py" \
  --pretrained "$ROOT/outputs/iclr27_phase4q/q0_long/checkpoint.pth" \
  --output_dir "$OUT" \
  --device cuda:0 \
  --dataset_file lvis_generated_img_seqs \
  --with_box_refine --two_stage \
  --batch_size "$BATCH" \
  --num_workers "$WORKERS" \
  --epochs "$EPOCHS" \
  --max_train_iters "$MAX_ITERS" \
  --ckpt_interval "$CKPT" \
  --save_period 1 \
  --tco_loss_coef 1.0 \
  --tco_alpha 0.5 \
  --score_mode base \
  --sample_mode random_interval \
  --sample_interval 1 \
  --max_len 250 \
  --merger_dropout 0 \
  --random_drop 0.1 \
  --fp_ratio 0.3 \
  --calculate_negative_samples \
  --sampler_lengths "${SAMPLER[@]}" \
  "${EXTRA_ARGS[@]}" \
  --seed "$((575700 + FOLD))" \
  --lr 2e-4 \
  --lr_backbone 0 \
  --lr_drop 100 \
  --filter_ignore
