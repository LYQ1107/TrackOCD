#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/iclr27_phase82p"
SHARDS="$OUT/features/q0_dinov2_shards"
LOGS="$OUT/logs/appearance"
mkdir -p "$SHARDS" "$LOGS" "$OUT/completion"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
GPUS=(4 5 6 7)
pids=(); names=()
for shard in 0 1 2 3; do
  out="$SHARDS/shard_$(printf '%02d' "$shard").npz"
  done_marker="$OUT/completion/appearance_shard_${shard}.done"
  launched="$OUT/completion/appearance_shard_${shard}.launched"
  if [[ -e "$done_marker" ]]; then
    continue
  fi
  if [[ -e "$launched" ]]; then
    echo "refusing to duplicate launched shard $shard" >&2
    exit 3
  fi
  tmp="$launched.$$.tmp"
  printf 'pid_pending supervisor=%s shard=%s gpu=%s utc=%s\n' "$$" "$shard" "${GPUS[$shard]}" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "$ROOT/scripts/iclr27_phase82p/extract_q0_appearance.py" \
    --shard "$shard" --num-shards 4 --device cuda:0 --batch 32 --out "$out" \
    > "$LOGS/shard_${shard}.log" 2>&1 &
  pids+=("$!"); names+=("$shard")
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    shard="${names[$i]}"
    out="$SHARDS/shard_$(printf '%02d' "$shard").npz"
    test -s "$out" && test -s "$out.json"
    tmp="$OUT/completion/appearance_shard_${shard}.done.$$.tmp"
    printf 'complete shard=%s utc=%s output=%s\n' "$shard" "$(date -u +%FT%TZ)" "$out" > "$tmp"
    mv "$tmp" "$OUT/completion/appearance_shard_${shard}.done"
  else
    echo "appearance shard ${names[$i]} failed; preserve .launched and log" >&2
    status=1
  fi
done
exit "$status"
