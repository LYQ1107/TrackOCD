#!/usr/bin/env bash
set -uo pipefail

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
if (( available_kib < safety_floor_kib )); then
  echo "RAM safety floor already crossed: available=${available_kib}KiB floor=${safety_floor_kib}KiB" >&2
  exit 61
fi
available_disk_kib=$(df --output=avail /data1 | tail -n 1)
if (( available_disk_kib < 41943040 )); then
  echo "less than 40 GiB available on /data1" >&2
  exit 62
fi

idle_gpus=()
while IFS=',' read -r raw_index raw_used raw_util; do
  index=${raw_index// /}; used=${raw_used// /}; util=${raw_util// /}
  if (( used <= 500 && util <= 10 )); then
    idle_gpus+=("$index")
  fi
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
if (( ${#idle_gpus[@]} < 1 )); then
  echo "need at least one genuinely idle GPU; found none" >&2
  exit 63
fi
worker_count=${#idle_gpus[@]}
if (( worker_count > 4 )); then worker_count=4; fi
selected_gpus=("${idle_gpus[@]:0:$worker_count}")
echo "selected physical GPUs: ${selected_gpus[*]}"
echo "RSS estimate: <=2 GiB/worker, <=8 GiB total; RAM safety floor ${safety_floor_kib} KiB"

for (( batch_start=0; batch_start<4; batch_start+=worker_count )); do
  pids=()
  names=()
  for (( offset=0; offset<worker_count && batch_start+offset<4; offset++ )); do
    fold=$((batch_start + offset))
    unit="dstm_seed1801_fold${fold}"
    launched="outputs/iclr27_phase18/markers/${unit}.launched"
    done_marker="outputs/iclr27_phase18/markers/${unit}.done"
    if [[ -e "$done_marker" ]]; then
      echo "already complete: $unit"
      continue
    fi
    if [[ -e "$launched" ]]; then
      echo "refusing blind relaunch of $unit: launched marker exists" >&2
      exit 64
    fi
    gpu=${selected_gpus[$offset]}
    read -r used util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu" | tr -d ' ' | tr ',' ' ')
    if (( used > 500 || util > 10 )); then
      echo "GPU $gpu ceased to be idle before $unit: used=${used}MiB util=${util}%" >&2
      exit 67
    fi
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i "$gpu"
    : > "$launched"
    setsid env CUDA_VISIBLE_DEVICES="$gpu" "$phase18_python" -m src.iclr27_phase18.training.train_dstm_fold \
      --fold "$fold" --seed 1801 --variant dstm --amp bf16 --device 0 \
      --best "outputs/iclr27_phase18/checkpoints/${unit}_best.pt" \
      --latest "outputs/iclr27_phase18/checkpoints/${unit}_latest.pt" \
      --summary "outputs/iclr27_phase18/eval/${unit}_training.json" \
      --done "$done_marker" \
      > "outputs/iclr27_phase18/logs/${unit}.log" 2>&1 &
    pids+=("$!"); names+=("$unit")
  done

  sleep 20
  available_after_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  if (( available_after_kib < safety_floor_kib )); then
    echo "post-launch RAM floor crossed: available=${available_after_kib}KiB" >&2
    for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    exit 65
  fi
  echo "post-launch MemAvailable KiB: $available_after_kib"
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
    exit 66
  fi
done

"$phase18_python" - <<'PY'
import json
from pathlib import Path
root=Path('outputs/iclr27_phase18')
for fold in range(4):
    unit=f'dstm_seed1801_fold{fold}'
    summary=root/'eval'/f'{unit}_training.json'
    best=root/'checkpoints'/f'{unit}_best.pt'
    latest=root/'checkpoints'/f'{unit}_latest.pt'
    done=root/'markers'/f'{unit}.done'
    assert summary.stat().st_size > 0 and best.stat().st_size > 0 and latest.stat().st_size > 0 and done.exists()
    value=json.loads(summary.read_text())
    assert value['updates'] >= 20000 and value['complete_unique_fit_row_passes'] >= 10
    assert value['finite_gradient_steps'] == value['updates'] and value['full_registered_run']
print('validated four complete registered DSTM fold runs')
PY

du -sh outputs/iclr27_phase18
echo PHASE18_MAIN_CROSSFIT_COMPLETE
