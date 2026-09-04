#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase82r"
TAG="${1:-balanced_formal}"
EPOCHS="${2:-15}"
mkdir -p "$OUT/completion" "$OUT/logs/balanced_residual"
GPUS=(4 5 6 7)
pids=(); folds=()
for fold in 0 1 2 3; do
  done_marker="$OUT/completion/${TAG}_f${fold}.done"
  launched="$OUT/completion/${TAG}_f${fold}.launched"
  [[ -e "$done_marker" ]] && continue
  [[ -e "$launched" ]] && { echo "refusing duplicate balanced fold $fold" >&2; exit 3; }
  tmp="$launched.$$.tmp"
  printf 'pid_pending supervisor=%s fold=%s gpu=%s tag=%s utc=%s\n' "$$" "$fold" "${GPUS[$fold]}" "$TAG" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PY" "$ROOT/scripts/iclr27_phase82r/train_balanced_residual.py" --fold "$fold" --device cuda:0 --tag "$TAG" --epochs "$EPOCHS" --batch-size 128 --checkpoint-interval 500 > "$OUT/logs/balanced_residual/${TAG}_f${fold}.log" 2>&1 &
  pids+=("$!"); folds+=("$fold")
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    fold="${folds[$i]}"
    test -s "$OUT/metrics/$TAG/fold${fold}_final.json"
    test -s "$OUT/checkpoints/$TAG/fold${fold}/latest.pt"
    done_marker="$OUT/completion/${TAG}_f${fold}.done"
    tmp="$done_marker.$$.tmp"
    printf 'complete tag=%s fold=%s utc=%s\n' "$TAG" "$fold" "$(date -u +%FT%TZ)" > "$tmp"
    mv "$tmp" "$done_marker"
  else
    echo "balanced fold ${folds[$i]} failed; preserve .launched and log" >&2
    status=1
  fi
done
exit "$status"
