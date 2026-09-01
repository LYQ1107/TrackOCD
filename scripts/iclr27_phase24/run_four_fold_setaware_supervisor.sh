#!/usr/bin/env bash
set -u
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
mkdir -p "$ROOT/outputs/iclr27_phase24/logs" "$ROOT/outputs/iclr27_phase24/completion"
declare -a pids=()
declare -a folds=(0 1 2 3)
declare -a gpus=(4 5 6 7)
for i in "${!folds[@]}"; do
  f="${folds[$i]}"; g="${gpus[$i]}"
  done_marker="$ROOT/outputs/iclr27_phase24/completion/setaware_f${f}.done"
  launch_marker="$ROOT/outputs/iclr27_phase24/completion/setaware_f${f}.launched"
  if [[ -e "$done_marker" ]]; then
    echo "fold $f already complete; supervisor will not relaunch"
    continue
  fi
  if [[ -e "$launch_marker" ]]; then
    echo "fold $f has a prior launch marker without completion; refusing blind relaunch" >&2
    exit 2
  fi
  env CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase24/train_set_selector.py" --fold "$f" --device cuda:0 --expected-physical-gpu "$g" --steps 4000 --batch-size 32 --checkpoint-every 500 --tag setaware >"$ROOT/outputs/iclr27_phase24/logs/setaware_f${f}.stdout.log" 2>"$ROOT/outputs/iclr27_phase24/logs/setaware_f${f}.stderr.log" &
  pids+=("$!")
  echo "fold=$f gpu=$g pid=${pids[-1]}"
done
printf '%s\n' "${pids[@]}" >"$ROOT/outputs/iclr27_phase24/logs/setaware_supervisor_pids.txt"
status=0
for p in "${pids[@]}"; do
  wait "$p" || status=$?
done
exit "$status"
