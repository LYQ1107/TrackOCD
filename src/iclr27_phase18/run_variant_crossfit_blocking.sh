#!/usr/bin/env bash
set -uo pipefail

if (( $# != 3 )); then echo "usage: $0 PREFIX VARIANT SEED" >&2; exit 2; fi
prefix=$1
variant=$2
seed=$3
phase18_root=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
phase18_python=/home/lwr/anaconda3/envs/AVI/bin/python
cd "$phase18_root"

nvidia-smi
free -h
df -h /data1
ps -e --no-headers | wc -l
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
total_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
safety_floor_kib=$((total_kib / 4))
if (( available_kib < safety_floor_kib )); then exit 81; fi
if (( $(df --output=avail /data1 | tail -n 1) < 41943040 )); then exit 82; fi

idle_gpus=()
while IFS=',' read -r raw_index raw_used raw_util; do
  index=${raw_index// /}; used=${raw_used// /}; util=${raw_util// /}
  if (( used <= 500 && util <= 10 )); then idle_gpus+=("$index"); fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
if (( ${#idle_gpus[@]} < 4 )); then
  echo "four-fold parallel block requires four genuinely idle GPUs; found ${#idle_gpus[@]}: ${idle_gpus[*]}" >&2
  exit 83
fi
gpus=("${idle_gpus[@]:0:4}")
echo "selected physical GPUs for $prefix: ${gpus[*]}"
echo "RSS estimate <=2 GiB/worker, <=8 GiB total; 25% RAM floor ${safety_floor_kib} KiB"

pids=(); names=()
for fold in 0 1 2 3; do
  unit="${prefix}_fold${fold}"
  launched="outputs/iclr27_phase18/markers/${unit}.launched"
  done_marker="outputs/iclr27_phase18/markers/${unit}.done"
  if [[ -e "$done_marker" ]]; then echo "already complete: $unit"; continue; fi
  if [[ -e "$launched" ]]; then echo "refusing blind relaunch: $unit" >&2; exit 84; fi
  gpu=${gpus[$fold]}
  read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ' | tr ',' ' ')
  if (( used > 500 || util > 10 )); then echo "GPU $gpu ceased to be idle" >&2; exit 85; fi
  : > "$launched"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" "$phase18_python" -m src.iclr27_phase18.training.train_dstm_fold \
    --fold "$fold" --seed "$seed" --variant "$variant" --amp bf16 --device 0 \
    --best "outputs/iclr27_phase18/checkpoints/${unit}_best.pt" \
    --latest "outputs/iclr27_phase18/checkpoints/${unit}_latest.pt" \
    --summary "outputs/iclr27_phase18/eval/${unit}_training.json" \
    --done "$done_marker" > "outputs/iclr27_phase18/logs/${unit}.log" 2>&1 &
  pids+=("$!"); names+=("$unit")
done

sleep 20
available_after_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_after_kib < safety_floor_kib )); then
  echo "post-launch RAM floor crossed" >&2
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  exit 86
fi
echo "post-launch MemAvailable KiB: $available_after_kib"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "completed ${names[$i]}"; else rc=$?; echo "failed ${names[$i]} rc=$rc" >&2; status=1; fi
done
if (( status != 0 )); then exit 87; fi

"$phase18_python" - "$prefix" <<'PY'
import json,sys
from pathlib import Path
prefix=sys.argv[1];root=Path('outputs/iclr27_phase18')
for fold in range(4):
    unit=f'{prefix}_fold{fold}'
    value=json.loads((root/'eval'/f'{unit}_training.json').read_text())
    assert (root/'markers'/f'{unit}.done').exists()
    assert (root/'checkpoints'/f'{unit}_best.pt').stat().st_size>0
    assert value['updates']>=20000 and value['complete_unique_fit_row_passes']>=10
    assert value['finite_gradient_steps']==value['updates'] and value['full_registered_run']
print(f'validated complete four-fold block: {prefix}')
PY
du -sh outputs/iclr27_phase18
echo "PHASE18_VARIANT_CROSSFIT_COMPLETE $prefix"
