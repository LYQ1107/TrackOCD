#!/usr/bin/env bash
# One bounded, blocking four-fold Phase22 supervisor.  Each fold owns one
# explicitly selected GPU; no broad process matching or background relaunch.
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
mkdir -p "$ROOT/outputs/iclr27_phase22/logs"
echo "[phase22] preflight $(date --iso-8601=seconds)"
free -h
ps -eo stat= | wc -l
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /data1 | tail -n 2

declare -a PIDS=()
declare -a FOLDS=(0 1 2 3)
declare -a GPUS=(0 1 2 3)
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "[phase22] stopping task-owned pid $pid"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM

for j in "${!FOLDS[@]}"; do
  fold=${FOLDS[$j]}; gpu=${GPUS[$j]}
  done_marker="$ROOT/outputs/iclr27_phase22/completion/f${fold}.done"
  launch_marker="$ROOT/outputs/iclr27_phase22/completion/f${fold}.launched"
  if [[ -f "$done_marker" ]]; then
    echo "[phase22] fold $fold already complete; skip"
    continue
  fi
  if [[ -f "$launch_marker" ]]; then
    echo "[phase22] refusing blind relaunch: $launch_marker exists"
    exit 2
  fi
  log="$ROOT/outputs/iclr27_phase22/logs/train_f${fold}.log"
  echo "[phase22] launch fold=$fold gpu=$gpu log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u "$ROOT/scripts/iclr27_phase22/train_proposal_refiner.py" \
    --fold "$fold" --device cuda:0 --steps 2000 --batch-size 256 --checkpoint-every 500 \
    --seed 20260828 >"$log" 2>&1 &
  PIDS+=("$!")
done

# One post-launch headroom check, then one blocking wait for all workers.
sleep 3
echo "[phase22] post-launch headroom"
free -h
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

status=0
for pid in "${PIDS[@]:-}"; do
  if ! wait "$pid"; then status=1; fi
done
trap - INT TERM
echo "[phase22] workers complete status=$status"
exit "$status"
