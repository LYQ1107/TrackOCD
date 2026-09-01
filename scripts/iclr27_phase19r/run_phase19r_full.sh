#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
UPDATES="${PHASE19R_UPDATES:-50000}"
BATCH="${PHASE19R_BATCH:-24}"
SEED="${PHASE19R_SEED:-1902}"
AMP="${PHASE19R_AMP:-bf16}"
GPUS_CSV="${PHASE19R_GPUS:-1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
if [ "${#GPUS[@]}" -ne 4 ]; then echo "exactly four GPU slots are required" >&2; exit 2; fi

mkdir -p outputs/iclr27_phase19r/{checkpoints,logs,completion,metrics}
read -r total_k avail_k _ < <(free -k | awk '/^Mem:/ {print $2, $7, $3}')
if [ "$avail_k" -lt $((total_k / 4)) ]; then echo "insufficient memory headroom before launch" >&2; exit 3; fi
for fold in 0 1 2 3; do
  done_marker="outputs/iclr27_phase19r/completion/fold${fold}.done"
  launched_marker="outputs/iclr27_phase19r/completion/fold${fold}.launched"
  if [ -e "$done_marker" ]; then continue; fi
  if [ -e "$launched_marker" ]; then echo "unfinished launched marker exists for fold ${fold}; refusing blind relaunch" >&2; exit 4; fi
  tmp="${launched_marker}.tmp"; printf 'gpu=%s seed=%s updates=%s\n' "${GPUS[$fold]}" "$SEED" "$UPDATES" > "$tmp"; mv "$tmp" "$launched_marker"
  CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" PYTHONPATH=. /home/lwr/anaconda3/envs/AVI/bin/python -m src.iclr27_phase19r.training.train_controller \
    --fold "$fold" --seed "$SEED" --updates "$UPDATES" --batch-size "$BATCH" \
    --device cuda:0 --amp "$AMP" --ladder L2 --max-states 16 --validation-episodes 64 \
    --best "outputs/iclr27_phase19r/checkpoints/fold${fold}_best_internal.pt" \
    --latest "outputs/iclr27_phase19r/checkpoints/fold${fold}_latest_valid.pt" \
    --summary "outputs/iclr27_phase19r/metrics/fold${fold}_training.json" \
    --done "$done_marker" > "outputs/iclr27_phase19r/logs/fold${fold}.log" 2>&1 &
done

# One supervisor wait: workers are bounded to four and each writes its own
# atomic completion marker.  We do not poll from the agent layer.
wait
for fold in 0 1 2 3; do test -f "outputs/iclr27_phase19r/completion/fold${fold}.done"; done
printf '%s\n' '{"complete":true,"folds":4}'
