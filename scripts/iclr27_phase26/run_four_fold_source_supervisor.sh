#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
mkdir -p "$ROOT/outputs/iclr27_phase26/logs" "$ROOT/outputs/iclr27_phase26/completion"
declare -a pids=()
declare -a launched=()
for spec in "0 4" "1 5" "2 6" "3 7"; do
  read -r fold gpu <<<"$spec"
  done="$ROOT/outputs/iclr27_phase26/completion/source_f${fold}.done"
  marker="$ROOT/outputs/iclr27_phase26/completion/source_f${fold}.launched"
  if [[ -f "$done" ]]; then echo "fold${fold}: done; skip"; continue; fi
  if [[ -f "$marker" ]]; then echo "fold${fold}: launched marker exists; skip (resume requires explicit audit)"; continue; fi
  echo "fold${fold}: launching physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u "$ROOT/scripts/iclr27_phase26/train_source_generator.py" --fold "$fold" --device cuda:0 --expected-physical-gpu "$gpu" --steps 2000 --batch-size 64 --checkpoint-every 500 --tag source >"$ROOT/outputs/iclr27_phase26/logs/source_f${fold}.stdout.log" 2>&1 &
  pids+=("$!"); launched+=("fold${fold}:gpu${gpu}:pid$!")
done
printf '%s\n' "${launched[@]:-none}" > "$ROOT/outputs/iclr27_phase26/logs/source_supervisor_launches.txt"
if ((${#pids[@]})); then
  wait "${pids[@]}"
fi
echo "source supervisor completed"
