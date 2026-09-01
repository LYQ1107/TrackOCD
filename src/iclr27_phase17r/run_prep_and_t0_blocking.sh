#!/usr/bin/env bash
set -uo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/AVI/bin/python
cd "$ROOT"

nvidia-smi
free -h
df -h /data1
ps -e --no-headers | wc -l

for gpu in 4 5 6 7; do
  read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ' | tr ',' ' ')
  if (( used > 500 || util > 10 )); then
    echo "GPU $gpu is not idle: used=${used}MiB util=${util}%" >&2
    exit 41
  fi
done

mkdir -p outputs/iclr27_phase17r/features/dinov3_shards outputs/iclr27_phase17r/logs outputs/iclr27_phase17r/markers outputs/iclr27_phase17r/checkpoints
pids=()
names=()

for shard in 0 1 2; do
  gpu=$((4 + shard))
  launched="outputs/iclr27_phase17r/markers/dinov3_shard_${shard}.launched"
  done_marker="outputs/iclr27_phase17r/markers/dinov3_shard_${shard}.done"
  if [[ -e "$done_marker" ]]; then
    continue
  fi
  if [[ -e "$launched" ]]; then
    echo "refusing blind relaunch of shard $shard with launched marker" >&2
    exit 42
  fi
  : > "$launched"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m src.iclr27_phase17r.representation.extract_dinov3_full \
    --shard-id "$shard" --num-shards 3 --device 0 --batch 96 \
    --out "outputs/iclr27_phase17r/features/dinov3_shards/shard_0${shard}.npz" \
    --meta "outputs/iclr27_phase17r/features/dinov3_shards/shard_0${shard}.json" \
    --done "$done_marker" \
    > "outputs/iclr27_phase17r/logs/dinov3_shard_0${shard}.log" 2>&1 &
  pids+=("$!"); names+=("dinov3_shard_${shard}")
done

t0_launched=outputs/iclr27_phase17r/markers/t0_full.launched
t0_done=outputs/iclr27_phase17r/markers/t0_full.done
if [[ ! -e "$t0_done" ]]; then
  if [[ -e "$t0_launched" ]]; then
    echo "refusing blind relaunch of T0 with launched marker" >&2
    exit 43
  fi
  : > "$t0_launched"
  CUDA_VISIBLE_DEVICES=7 "$PY" -m src.iclr27_phase17r.training.train_full_model \
    --variant t0 --model-name T0_FULL_DINOV2_OBSERVABILITY_SEMANTIC \
    --features data/iclr27_phase17r/sources/public_dinov2_cls_roi.npz \
    --updates 6000 --global-batch 64 --checkpoint-interval 1000 \
    --best outputs/iclr27_phase17r/checkpoints/t0_best.pt \
    --latest outputs/iclr27_phase17r/checkpoints/t0_latest.pt \
    --summary outputs/iclr27_phase17r/eval/t0_training_summary.json \
    --done "$t0_done" \
    > outputs/iclr27_phase17r/logs/t0_training.log 2>&1 &
  pids+=("$!"); names+=("t0_full")
fi

sleep 20
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < 33554432 )); then
  echo "memory safety floor crossed: ${available_kib} KiB available" >&2
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 44
fi
echo "post-launch memory available KiB: $available_kib"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "completed ${names[$i]}"
  else
    rc=$?
    echo "failed ${names[$i]} rc=$rc" >&2
    status=1
  fi
done
if (( status != 0 )); then
  exit 45
fi

if [[ ! -e outputs/iclr27_phase17r/markers/dinov3_full.done ]]; then
  : > outputs/iclr27_phase17r/markers/dinov3_merge.launched
  "$PY" -m src.iclr27_phase17r.representation.merge_dinov3_full \
    --shards outputs/iclr27_phase17r/features/dinov3_shards/shard_00.npz \
             outputs/iclr27_phase17r/features/dinov3_shards/shard_01.npz \
             outputs/iclr27_phase17r/features/dinov3_shards/shard_02.npz \
    --out outputs/iclr27_phase17r/features/full_public_dinov3.npz \
    --meta outputs/iclr27_phase17r/features/full_public_dinov3.json \
    --done outputs/iclr27_phase17r/markers/dinov3_full.done
fi

test -s outputs/iclr27_phase17r/features/full_public_dinov3.npz
test -s outputs/iclr27_phase17r/checkpoints/t0_best.pt
test -e outputs/iclr27_phase17r/markers/t0_full.done
echo "PHASE17R_PREP_AND_T0_COMPLETE"
