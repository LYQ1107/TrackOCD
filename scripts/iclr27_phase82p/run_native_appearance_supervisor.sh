#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/outputs/iclr27_phase82p"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
TAG="${1:-native_r1}"
GPUS=(4 5 6 7)
mkdir -p "$OUT/features/native_dinov2_shards" "$OUT/logs/native_appearance" "$OUT/completion"
pids=(); shards=()
for shard in 0 1 2 3; do
  out="$OUT/features/native_dinov2_shards/${TAG}_shard_$(printf '%02d' "$shard").npz"
  done_marker="$OUT/completion/native_appearance_${TAG}_shard_${shard}.done"
  launched="$OUT/completion/native_appearance_${TAG}_shard_${shard}.launched"
  [[ -e "$done_marker" ]] && continue
  [[ -e "$launched" ]] && { echo "refusing duplicate native shard $shard" >&2; exit 3; }
  tmp="$launched.$$.tmp"
  printf 'pid_pending supervisor=%s shard=%s gpu=%s tag=%s utc=%s\n' "$$" "$shard" "${GPUS[$shard]}" "$TAG" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "$ROOT/scripts/iclr27_phase82p/extract_native_appearance.py" --shard "$shard" --num-shards 4 --device cuda:0 --batch 32 --out "$out" > "$OUT/logs/native_appearance/${TAG}_shard_${shard}.log" 2>&1 &
  pids+=("$!"); shards+=("$shard")
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    shard="${shards[$i]}"
    out="$OUT/features/native_dinov2_shards/${TAG}_shard_$(printf '%02d' "$shard").npz"
    test -s "$out"; test -s "$out.json"
    tmp="$OUT/completion/native_appearance_${TAG}_shard_${shard}.done.$$.tmp"
    printf 'complete tag=%s shard=%s utc=%s\n' "$TAG" "$shard" "$(date -u +%FT%TZ)" > "$tmp"
    mv "$tmp" "$OUT/completion/native_appearance_${TAG}_shard_${shard}.done"
  else
    echo "native appearance shard ${shards[$i]} failed; preserve marker" >&2
    status=1
  fi
done
exit "$status"
