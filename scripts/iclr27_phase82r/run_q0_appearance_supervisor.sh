#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase82r"
TAG="${1:-q0_corrected}"
mkdir -p "$OUT/features/q0_dinov2_shards" "$OUT/logs/q0_appearance" "$OUT/completion"
GPUS=(4 5 6 7)
pids=(); shards=()
for shard in 0 1 2 3; do
  out="$OUT/features/q0_dinov2_shards/${TAG}_shard_$(printf '%02d' "$shard").npz"
  done_marker="$OUT/completion/q0_appearance_${TAG}_shard_${shard}.done"
  launched="$OUT/completion/q0_appearance_${TAG}_shard_${shard}.launched"
  [[ -e "$done_marker" ]] && continue
  [[ -e "$launched" ]] && { echo "refusing duplicate q0 appearance shard $shard" >&2; exit 3; }
  tmp="$launched.$$.tmp"
  printf 'pid_pending supervisor=%s shard=%s gpu=%s tag=%s utc=%s\n' "$$" "$shard" "${GPUS[$shard]}" "$TAG" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "$ROOT/scripts/iclr27_phase82r/extract_q0_appearance.py" --shard "$shard" --num-shards 4 --device cuda:0 --batch 32 --tag "$TAG" --out "$out" > "$OUT/logs/q0_appearance/${TAG}_shard_${shard}.log" 2>&1 &
  pids+=("$!"); shards+=("$shard")
done
status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    shard="${shards[$i]}"
    out="$OUT/features/q0_dinov2_shards/${TAG}_shard_$(printf '%02d' "$shard").npz"
    test -s "$out"; test -s "$out.json"
    done_marker="$OUT/completion/q0_appearance_${TAG}_shard_${shard}.done"
    tmp="$OUT/completion/q0_appearance_${TAG}_shard_${shard}.done.$$.tmp"
    printf 'complete tag=%s shard=%s utc=%s\n' "$TAG" "$shard" "$(date -u +%FT%TZ)" > "$tmp"
    mv "$tmp" "$done_marker"
  else
    echo "q0 appearance shard ${shards[$i]} failed; preserve marker" >&2
    status=1
  fi
done
exit "$status"
