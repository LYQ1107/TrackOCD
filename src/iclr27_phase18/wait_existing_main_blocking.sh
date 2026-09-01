#!/usr/bin/env bash
set -uo pipefail

phase18_root=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
phase18_python=/home/lwr/anaconda3/envs/AVI/bin/python
cd "$phase18_root"

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
free -h
df -h /data1
ps -e --no-headers | wc -l

pids=()
for fold in 0 1 2 3; do
  if [[ -e "outputs/iclr27_phase18/markers/dstm_seed1801_fold${fold}.done" ]]; then
    continue
  fi
  mapfile -t found < <(pgrep -f "[s]rc.iclr27_phase18.training.train_dstm_fold --fold ${fold} --seed 1801 --variant dstm" || true)
  if (( ${#found[@]} != 1 )); then
    echo "expected exactly one live Phase18 process for fold $fold, found ${#found[@]}: ${found[*]}" >&2
    exit 71
  fi
  cmd=$(tr '\0' ' ' < "/proc/${found[0]}/cmdline")
  if [[ "$cmd" != *"--done outputs/iclr27_phase18/markers/dstm_seed1801_fold${fold}.done"* ]]; then
    echo "resolved PID ${found[0]} has unexpected command: $cmd" >&2
    exit 72
  fi
  pids+=("${found[0]}")
done
echo "waiting for exact existing PIDs: ${pids[*]}"

while true; do
  live=0
  for pid in "${pids[@]}"; do
    if [[ -r "/proc/$pid/stat" ]]; then
      state=$(awk '{print $3}' "/proc/$pid/stat")
      if [[ "$state" != "Z" ]]; then live=$((live + 1)); fi
    fi
  done
  if (( live == 0 )); then break; fi
  sleep 30
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
    assert summary.stat().st_size > 0 and best.stat().st_size > 0 and latest.stat().st_size > 0 and done.exists(), unit
    value=json.loads(summary.read_text())
    assert value['updates'] >= 20000 and value['complete_unique_fit_row_passes'] >= 10, unit
    assert value['finite_gradient_steps'] == value['updates'] and value['full_registered_run'], unit
print('validated four complete registered DSTM fold runs after terminal-session recovery')
PY

for fold in 0 1 2 3; do tail -n 3 "outputs/iclr27_phase18/logs/dstm_seed1801_fold${fold}.log"; done
du -sh outputs/iclr27_phase18
echo PHASE18_MAIN_CROSSFIT_RECOVERED_COMPLETE
