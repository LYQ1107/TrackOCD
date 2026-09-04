#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
ARCHIVE="/data2/usr_for_deadline/trackocd_phase83"
OUT="$ROOT/outputs/iclr27_phase83"
TAG="${1:-a2_corrected_r1}"
NATIVE="$ARCHIVE/a2_full/native_lineage.jsonl"
FRAMES="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames"
FEATURE_DIR="$ARCHIVE/$TAG"
mkdir -p "$FEATURE_DIR" "$OUT/completion" "$OUT/logs/a2_appearance"
[[ -s "$NATIVE" ]] || { echo "missing A2 native lineage: $NATIVE" >&2; exit 2; }
# GPU4 became occupied by an unrelated job after the initial preflight.  Use
# the four idle cards 5--8; CUDA_VISIBLE_DEVICES remaps each worker to cuda:0.
GPUS=(5 6 7 8)
pids=(); shards=()
for shard in 0 1 2 3; do
  out="$FEATURE_DIR/shard_$(printf '%02d' "$shard").npz"
  marker="$OUT/completion/a2_appearance_${TAG}_shard_${shard}"
  [[ -e "$marker.done" ]] && continue
  [[ -e "$marker.launched" ]] && { echo "refusing duplicate shard $shard" >&2; exit 3; }
  tmp="$marker.launched.$$.tmp"
  printf 'pid_pending supervisor=%s shard=%s gpu=%s tag=%s\n' "$$" "$shard" "${GPUS[$shard]}" "$TAG" > "$tmp"
  mv "$tmp" "$marker.launched"
  CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" "$PY" "$ROOT/scripts/iclr27_phase83/extract_a2_appearance.py" --native "$NATIVE" --frames "$FRAMES" --shard "$shard" --num-shards 4 --device cuda:0 --batch 32 --out "$out" > "$OUT/logs/a2_appearance/${TAG}_shard_${shard}.log" 2>&1 &
  pids+=("$!"); shards+=("$shard")
done
status=0
for i in "${!pids[@]}"; do
  shard="${shards[$i]}"; marker="$OUT/completion/a2_appearance_${TAG}_shard_${shard}"
  if wait "${pids[$i]}"; then
    out="$FEATURE_DIR/shard_$(printf '%02d' "$shard").npz"
    if [[ ! -s "$out" || ! -s "$out.json" ]]; then echo "missing shard artifact $shard" >&2; status=1; continue; fi
    tmp="$marker.done.$$.tmp"
    printf 'complete tag=%s shard=%s utc=%s\n' "$TAG" "$shard" "$(date -u +%FT%TZ)" > "$tmp"
    mv "$tmp" "$marker.done"
  else
    echo "A2 appearance shard $shard failed; retaining .launched" >&2
    status=1
  fi
done
exit "$status"
