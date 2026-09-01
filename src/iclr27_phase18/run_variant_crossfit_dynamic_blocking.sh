#!/usr/bin/env bash
set -uo pipefail

if (( $# != 3 )); then echo "usage: $0 PREFIX VARIANT SEED" >&2; exit 2; fi
prefix=$1
variant=$2
seed=$3
phase18_root=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
phase18_python=/home/lwr/anaconda3/envs/AVI/bin/python
cd "$phase18_root"

# This supervisor is intentionally dynamic: it never shares an occupied GPU,
# and fills at most four slots when genuinely idle devices become available.
nvidia-smi
free -h
df -h /data1
ps -e --no-headers | wc -l

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
total_kib=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
safety_floor_kib=$((total_kib / 4))
if (( available_kib < safety_floor_kib )); then exit 81; fi
if (( $(df --output=avail /data1 | tail -n 1) < 41943040 )); then exit 82; fi
echo "RSS estimate <=2 GiB/worker, <=8 GiB total; 25% RAM floor ${safety_floor_kib} KiB"

pending=()
for fold in 0 1 2 3; do
  unit="${prefix}_fold${fold}"
  launched="outputs/iclr27_phase18/markers/${unit}.launched"
  done_marker="outputs/iclr27_phase18/markers/${unit}.done"
  if [[ -e "$done_marker" ]]; then
    echo "already complete: $unit"
  elif [[ -e "$launched" ]]; then
    echo "refusing blind relaunch: $unit" >&2
    exit 84
  else
    pending+=("$fold")
  fi
done

declare -A unit_for_pid=()
declare -A gpu_for_pid=()
running_pids=()
failure=0
cleanup() {
  local rc=$?
  if (( rc != 0 || failure != 0 )); then
    for pid in "${running_pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    done
    for pid in "${running_pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  fi
}
trap cleanup EXIT INT TERM

last_wait_report=0
while (( ${#pending[@]} > 0 || ${#running_pids[@]} > 0 )); do
  survivors=()
  for pid in "${running_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      survivors+=("$pid")
      continue
    fi
    if wait "$pid"; then
      echo "completed ${unit_for_pid[$pid]} on GPU ${gpu_for_pid[$pid]}"
    else
      rc=$?
      echo "failed ${unit_for_pid[$pid]} on GPU ${gpu_for_pid[$pid]} rc=$rc" >&2
      failure=1
    fi
    unset 'unit_for_pid[$pid]' 'gpu_for_pid[$pid]'
  done
  running_pids=("${survivors[@]}")
  if (( failure != 0 )); then exit 87; fi

  used_by_us=" "
  for pid in "${running_pids[@]}"; do used_by_us+="${gpu_for_pid[$pid]} "; done

  while (( ${#pending[@]} > 0 && ${#running_pids[@]} < 4 )); do
    selected_gpu=""
    while IFS=',' read -r raw_index raw_used raw_util; do
      index=${raw_index// /}; used=${raw_used// /}; util=${raw_util// /}
      if [[ "$used_by_us" == *" $index "* ]]; then continue; fi
      if (( used <= 500 && util <= 10 )); then selected_gpu=$index; break; fi
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    if [[ -z "$selected_gpu" ]]; then break; fi

    fold=${pending[0]}
    pending=("${pending[@]:1}")
    unit="${prefix}_fold${fold}"
    launched="outputs/iclr27_phase18/markers/${unit}.launched"
    done_marker="outputs/iclr27_phase18/markers/${unit}.done"

    # Recheck the exact device immediately before the training process starts.
    read -r check_used check_util < <(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$selected_gpu" | tr -d ' ' | tr ',' ' ')
    if (( check_used > 500 || check_util > 10 )); then
      pending=("$fold" "${pending[@]}")
      continue
    fi
    marker_tmp="${launched}.tmp.$$"
    printf 'unit=%s\ngpu=%s\nseed=%s\nvariant=%s\n' "$unit" "$selected_gpu" "$seed" "$variant" > "$marker_tmp"
    mv "$marker_tmp" "$launched"
    setsid env CUDA_VISIBLE_DEVICES="$selected_gpu" "$phase18_python" -m src.iclr27_phase18.training.train_dstm_fold \
      --fold "$fold" --seed "$seed" --variant "$variant" --amp bf16 --device 0 \
      --best "outputs/iclr27_phase18/checkpoints/${unit}_best.pt" \
      --latest "outputs/iclr27_phase18/checkpoints/${unit}_latest.pt" \
      --summary "outputs/iclr27_phase18/eval/${unit}_training.json" \
      --done "$done_marker" > "outputs/iclr27_phase18/logs/${unit}.log" 2>&1 &
    pid=$!
    running_pids+=("$pid")
    unit_for_pid[$pid]=$unit
    gpu_for_pid[$pid]=$selected_gpu
    used_by_us+="$selected_gpu "
    echo "launched $unit pid=$pid on genuinely idle GPU $selected_gpu"

    sleep 10
    available_after_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    if (( available_after_kib < safety_floor_kib )); then
      echo "post-launch RAM floor crossed after $unit" >&2
      failure=1
      exit 86
    fi
    echo "post-launch MemAvailable KiB: $available_after_kib"
  done

  if (( ${#pending[@]} > 0 )); then
    now=$(date +%s)
    if (( now - last_wait_report >= 600 )); then
      echo "waiting for an idle GPU; pending folds: ${pending[*]}; running workers: ${#running_pids[@]}"
      last_wait_report=$now
    fi
  fi
  if (( ${#pending[@]} > 0 || ${#running_pids[@]} > 0 )); then sleep 20; fi
done

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
print(f'validated complete four-fold dynamic block: {prefix}')
PY
du -sh outputs/iclr27_phase18
echo "PHASE18_DYNAMIC_VARIANT_CROSSFIT_COMPLETE $prefix"
