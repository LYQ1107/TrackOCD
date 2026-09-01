#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase15s/features
mkdir -p "$OUT" "$ROOT/outputs/iclr27_phase15s/logs"
echo "[phase15s] feature preflight" | tee "$ROOT/outputs/iclr27_phase15s/logs/features_preflight.log"
df -h /data1 | tee -a "$ROOT/outputs/iclr27_phase15s/logs/features_preflight.log"
free -h | tee -a "$ROOT/outputs/iclr27_phase15s/logs/features_preflight.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$ROOT/outputs/iclr27_phase15s/logs/features_preflight.log"
ps -eo stat= | wc -l | tee -a "$ROOT/outputs/iclr27_phase15s/logs/features_preflight.log"
run_unit() {
  local tag=$1; shift
  local marker="$OUT/$tag.launched" done="$OUT/$tag.done"
  if [[ -f "$done" ]]; then echo "$tag already complete"; return; fi
  if [[ -f "$marker" ]]; then echo "$tag launched without completion; refusing blind relaunch"; exit 2; fi
  touch "$marker"
  PYTHONPATH="$ROOT" /home/lwr/anaconda3/envs/AVI/bin/python -m src.iclr27_phase15s.representation.extract_cls_roi "$@"
  touch "$done"
}
run_unit public \
  --proposals outputs/iclr27_phase15s/dsct_bank/public_roles/proposals.csv \
  --annotation data/iclr27_phase15s/sources/validation_public_roles.json \
  --out outputs/iclr27_phase15s/features/public_cls_roi.npz --device cuda:${PHASE15S_GPU:-2} --batch 16
run_unit devplus \
  --proposals data/iclr27_phase15s/sources/proposals_aligned.csv \
  --annotation data/iclr27_phase15s/sources/devplus_annotation.json \
  --out outputs/iclr27_phase15s/features/devplus_cls_roi.npz --device cuda:${PHASE15S_GPU:-2} --batch 16 \
  --reuse-cls data/iclr27_phase15s/sources/devplus_cls_features.npz
echo PHASE15S_FEATURES_DONE
