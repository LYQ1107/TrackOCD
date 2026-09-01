#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
OUT="$ROOT/outputs/iclr27_phase29"
mkdir -p "$OUT/logs" "$OUT/completion"

# Fixed bounded one-worker-per-fold schedule. CUDA_VISIBLE_DEVICES maps the
# selected physical card to cuda:0 inside each worker, and the worker asserts
# the expected physical GPU before touching data.
declare -a PIDS=()
declare -a FOLDS=(0 1 2 3)
declare -a GPUS=(4 5 6 7)
for i in "${!FOLDS[@]}"; do
  fold="${FOLDS[$i]}"; gpu="${GPUS[$i]}"
  done_marker="$OUT/completion/domain_aligned_f${fold}.done"
  launched_marker="$OUT/completion/domain_aligned_f${fold}.launched"
  if [[ -e "$done_marker" ]]; then
    echo "skip fold=$fold (done)"
    continue
  fi
  if [[ -e "$launched_marker" ]]; then
    echo "refuse relaunch fold=$fold (launched marker exists)" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/iclr27_phase29/train_domain_aligned.py" \
    --fold "$fold" --device cuda:0 --expected-physical-gpu "$gpu" \
    --steps 2000 --batch-size 32 --checkpoint-every 500 --tag domain_aligned \
    >"$OUT/logs/domain_aligned_f${fold}.stdout.log" 2>&1 &
  PIDS+=("$!")
done

# One blocking wait for all task-owned workers; no agent-level polling.
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
exit "$status"
