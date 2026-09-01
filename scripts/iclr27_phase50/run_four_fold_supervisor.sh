#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
TAG="${1:-e2e_formal}"
STEPS="${2:-1000}"
OUT="$ROOT/outputs/iclr27_phase50"
mkdir -p "$OUT/logs" "$OUT/completion" "$OUT/checkpoints" "$OUT/metrics"

declare -a pids=()
declare -a folds=()
declare -a gpus=(4 5 6 7)
for fold in 0 1 2 3; do
  run="${TAG}_f${fold}"
  done="$OUT/completion/${run}.done"
  launched="$OUT/completion/${run}.launched"
  if [[ -f "$done" ]]; then
    echo "skip completed $run"; continue
  fi
  if [[ -f "$launched" ]]; then
    echo "refusing to relaunch launched-but-incomplete unit: $launched" >&2
    exit 2
  fi
  gpu="${gpus[$fold]}"
  echo "launch fold=$fold physical_gpu=$gpu run=$run"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" OMP_NUM_THREADS=1 \
    "$PY" "$ROOT/scripts/iclr27_phase50/train_end_to_end.py" \
    --fold "$fold" --steps "$STEPS" --tag "$TAG" --device cuda:0 \
    --expected-physical-gpu "$gpu" --checkpoint-every 500 \
    >"$OUT/logs/${run}.log" 2>&1 &
  pids+=("$!"); folds+=("$fold")
done

# One bounded supervisor: at most four workers, then one blocking wait per PID.
sleep 2
free -h >"$OUT/logs/${TAG}_postlaunch_free.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >"$OUT/logs/${TAG}_postlaunch_gpu.txt"
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "fold ${folds[$i]} completed"
  else
    echo "fold ${folds[$i]} failed; see $OUT/logs/${TAG}_f${folds[$i]}.log" >&2
    status=1
  fi
done
if [[ "$status" -eq 0 ]]; then
  tmp="$OUT/completion/.${TAG}.supervisor.done.tmp"
  printf '{"phase":50,"tag":"%s","steps":%s,"folds":[0,1,2,3]}\n' "$TAG" "$STEPS" >"$tmp"
  mv -f "$tmp" "$OUT/completion/${TAG}.supervisor.done"
fi
exit "$status"
