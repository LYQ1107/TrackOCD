#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/iclr27_phase82p"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-residual_formal}"
STEPS="${2:-1000}"
GPUS=(4 5 6 7)
mkdir -p "$OUT/completion" "$OUT/logs/residual"
pids=(); folds=()
for fold in 0 1 2 3; do
  done_marker="$OUT/completion/${TAG}_f${fold}.done"
  launched="$OUT/completion/${TAG}_f${fold}.launched"
  if [[ -e "$done_marker" ]]; then continue; fi
  if [[ -e "$launched" ]]; then
    echo "refusing to duplicate launched fold $fold for tag $TAG" >&2
    exit 3
  fi
  tmp="$launched.$$.tmp"
  printf 'pid_pending supervisor=%s fold=%s gpu=%s tag=%s utc=%s\n' "$$" "$fold" "${GPUS[$fold]}" "$TAG" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$fold]}" "$PY" "$ROOT/scripts/iclr27_phase82p/train_residual.py" \
    --fold "$fold" --device cuda:0 --tag "$TAG" --steps "$STEPS" --batch-size 32 --checkpoint-interval 500 \
    > "$OUT/logs/residual/${TAG}_f${fold}.log" 2>&1 &
  pids+=("$!"); folds+=("$fold")
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    fold="${folds[$i]}"
    test -s "$OUT/metrics/$TAG/fold${fold}_final.json"
    test -s "$OUT/checkpoints/$TAG/fold${fold}/latest.pt"
    tmp="$OUT/completion/${TAG}_f${fold}.done.$$.tmp"
    printf 'complete tag=%s fold=%s utc=%s\n' "$TAG" "$fold" "$(date -u +%FT%TZ)" > "$tmp"
    mv "$tmp" "$OUT/completion/${TAG}_f${fold}.done"
  else
    echo "residual fold ${folds[$i]} failed; preserve .launched and log" >&2
    status=1
  fi
done
exit "$status"
