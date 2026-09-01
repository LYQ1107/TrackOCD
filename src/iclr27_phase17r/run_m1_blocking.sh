#!/usr/bin/env bash
set -uo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
TORCHRUN=/home/lwr/anaconda3/envs/AVI/bin/torchrun
cd "$ROOT"
nvidia-smi
free -h
df -h /data1
ps -e --no-headers | wc -l

for gpu in 4 5 6 8; do
  read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ' | tr ',' ' ')
  if (( used > 500 || util > 10 )); then
    echo "GPU $gpu is not idle: used=${used}MiB util=${util}%" >&2
    exit 81
  fi
done

launched=outputs/iclr27_phase17r/markers/m1_full.launched
done_marker=outputs/iclr27_phase17r/markers/m1_full.done
if [[ -e "$done_marker" || -e "$launched" ]]; then
  echo "refusing blind M1 relaunch" >&2
  exit 82
fi
: > "$launched"

setsid env CUDA_VISIBLE_DEVICES=4,5,6,8 "$TORCHRUN" --standalone --nproc_per_node=4 \
  -m src.iclr27_phase17r.training.train_full_model \
  --variant m1 --model-name M1_OBSERVABILITY_GATED_PQIR \
  --features outputs/iclr27_phase17r/features/full_public_dinov3.npz \
  --updates 12000 --global-batch 64 --checkpoint-interval 1000 --amp-dtype bf16 \
  --best outputs/iclr27_phase17r/checkpoints/m1_best.pt \
  --latest outputs/iclr27_phase17r/checkpoints/m1_latest.pt \
  --summary outputs/iclr27_phase17r/eval/main_training_summary.json \
  --done "$done_marker" \
  > outputs/iclr27_phase17r/logs/m1_training.log 2>&1 &
pid=$!

sleep 20
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 33554432 )); then
  echo "memory safety floor crossed: ${available_kib} KiB available" >&2
  kill -TERM -- "-$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  exit 83
fi
echo "post-launch memory available KiB: $available_kib"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

if ! wait "$pid"; then
  rc=$?
  echo "M1 failed rc=$rc" >&2
  exit 84
fi
test -e "$done_marker"
test -s outputs/iclr27_phase17r/checkpoints/m1_best.pt
test -s outputs/iclr27_phase17r/eval/main_training_summary.json
tail -n 20 outputs/iclr27_phase17r/logs/m1_training.log
echo PHASE17R_M1_COMPLETE
