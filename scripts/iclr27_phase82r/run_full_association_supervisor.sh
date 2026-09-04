#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TAG="${1:-full_assoc_formal_r1}"
EPOCHS="${2:-15}"
OUT="$ROOT/outputs/iclr27_phase82r"
COMP="$OUT/completion/$TAG"
LOG="$OUT/logs/$TAG"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
mkdir -p "$COMP" "$LOG"
GPUS=(4 5 6 7)
pids=()
for fold in 0 1 2 3; do
  done_marker="$COMP/fold${fold}.done"
  launched_marker="$COMP/fold${fold}.launched"
  if [[ -f "$done_marker" ]]; then
    echo "fold${fold}: existing done, skip"
    continue
  fi
  if [[ -f "$launched_marker" ]]; then
    echo "fold${fold}: already launched without done, skip (manual recovery required)"
    continue
  fi
  tmp="${launched_marker}.$$.$RANDOM.tmp"
  printf '{"fold":%d,"gpu":%d,"tag":"%s","launched_utc":"%s"}\n' "$fold" "${GPUS[$fold]}" "$TAG" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv "$tmp" "$launched_marker"
  CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PY" "$ROOT/scripts/iclr27_phase82r/train_full_association.py" --fold "$fold" --device cuda:0 --tag "$TAG" --epochs "$EPOCHS" > "$LOG/fold${fold}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
if [[ "$rc" -ne 0 ]]; then
  exit "$rc"
fi
for fold in 0 1 2 3; do
  [[ -f "$OUT/metrics/$TAG/fold${fold}_final.json" ]] || { echo "missing fold${fold} final metrics" >&2; exit 2; }
  tmp="$COMP/fold${fold}.done.$$.$RANDOM.tmp"
  printf '{"fold":%d,"tag":"%s","metrics":"%s","completed_utc":"%s"}\n' "$fold" "$TAG" "$OUT/metrics/$TAG/fold${fold}_final.json" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
  mv "$tmp" "$COMP/fold${fold}.done"
done
echo "completed tag=$TAG"
