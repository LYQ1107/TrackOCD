#!/usr/bin/env bash
set -uo pipefail
if (( $# != 3 )); then echo "usage: $0 PREFIX VARIANT SEED" >&2; exit 2; fi
prefix=$1
variant=$2
seed=$3
phase18_root=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
phase18_python=/home/lwr/anaconda3/envs/AVI/bin/python
cd "$phase18_root"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /data1
ps -e --no-headers | wc -l
pids=()
for fold in 0 1 2 3; do
  if [[ -e "outputs/iclr27_phase18/markers/${prefix}_fold${fold}.done" ]]; then continue; fi
  mapfile -t found < <(pgrep -f "[s]rc.iclr27_phase18.training.train_dstm_fold --fold ${fold} --seed ${seed} --variant ${variant}" || true)
  if (( ${#found[@]} != 1 )); then echo "expected one PID for ${prefix}_fold${fold}, found ${#found[@]}" >&2; exit 91; fi
  cmd=$(tr '\0' ' ' < "/proc/${found[0]}/cmdline")
  if [[ "$cmd" != *"--done outputs/iclr27_phase18/markers/${prefix}_fold${fold}.done"* ]]; then exit 92; fi
  pids+=("${found[0]}")
done
echo "waiting exact PIDs for $prefix: ${pids[*]}"
while true; do
  live=0
  for pid in "${pids[@]}"; do
    if [[ -r "/proc/$pid/stat" ]] && [[ $(awk '{print $3}' "/proc/$pid/stat") != Z ]]; then live=$((live+1)); fi
  done
  if (( live==0 )); then break; fi
  sleep 30
done
"$phase18_python" - "$prefix" <<'PY'
import json,sys
from pathlib import Path
prefix=sys.argv[1];root=Path('outputs/iclr27_phase18')
for fold in range(4):
    unit=f'{prefix}_fold{fold}'
    value=json.loads((root/'eval'/f'{unit}_training.json').read_text())
    assert (root/'markers'/f'{unit}.done').exists()
    assert value['updates']>=20000
    assert value['complete_unique_fit_row_passes']>=10 and value['finite_gradient_steps']==value['updates']
print(f'validated recovered four-fold block: {prefix}')
PY
for fold in 0 1 2 3; do tail -n 2 "outputs/iclr27_phase18/logs/${prefix}_fold${fold}.log"; done
echo "PHASE18_VARIANT_RECOVERED_COMPLETE $prefix"
