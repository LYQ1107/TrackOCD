#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/locatemot/bin/python"
GPUS=(4 5 6 7)
OUT="$ROOT/outputs/iclr27_phase25"
mkdir -p "$OUT/logs" "$OUT/completion" "$OUT/checkpoints" "$OUT/metrics"

# This supervisor is intentionally bounded to one worker per fold.  Each
# worker gets a single physical GPU through CUDA_VISIBLE_DEVICES and asserts
# that mapping before writing any checkpoint.
pids=()
for f in 0 1 2 3; do
  g="${GPUS[$f]}"
  done_marker="$OUT/completion/attention_f${f}.done"
  launch_marker="$OUT/completion/attention_f${f}.launched"
  if [[ -f "$done_marker" ]]; then
    continue
  fi
  if [[ -f "$launch_marker" ]]; then
    echo "refusing relaunch of fold $f with existing marker $launch_marker" >&2
    exit 2
  fi
  env CUDA_VISIBLE_DEVICES="$g" PYTHONPATH="$ROOT" "$PY" -u "$ROOT/scripts/iclr27_phase25/train_set_selector.py" \
    --fold "$f" --device cuda:0 --expected-physical-gpu "$g" --steps 4000 --batch-size 32 --checkpoint-every 500 --tag attention \
    >"$OUT/logs/attention_f${f}.stdout.log" 2>"$OUT/logs/attention_f${f}.stderr.log" &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" >"$OUT/logs/attention_supervisor_pids.txt"
status=0
for p in "${pids[@]}"; do
  if ! wait "$p"; then status=1; fi
done
exit "$status"
