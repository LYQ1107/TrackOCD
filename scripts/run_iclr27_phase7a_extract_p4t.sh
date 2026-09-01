#!/usr/bin/env bash
# Extract DINOv2 features for the Phase 4T subset on 3 idle GPUs.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
PY=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python
CSV=$ROOT/outputs/iclr27_phase7a/assets/p4t_dinov2_subset.csv
OUT=$ROOT/outputs/iclr27_phase7a/assets/p4t_dinov2
mkdir -p "$OUT"

run_shard() {
  local gpu=$1 start=$2 end=$3
  local tag=${start}-${end}
  if [[ -f "$OUT/shard_${tag}.done" ]]; then
    echo "shard $tag done"; return
  fi
  PYTHONPATH="$ROOT" CUDA_VISIBLE_DEVICES="$gpu" "$PY" \
    "$ROOT/src/iclr27_phase7a/data/extract_p4t_dinov2.py" \
    --csv "$CSV" --out "$OUT/shard_${tag}.npz" \
    --device cuda:0 --batch 96 --start "$start" --end "$end" \
    > "$OUT/shard_${tag}.log" 2>&1 &
  echo $! > "$OUT/shard_${tag}.pid"
}

run_shard 7 0 13016
run_shard 9 13016 26031
run_shard 0 26031 39047

for tag in 0-13016 13016-26031 26031-39047; do
  while kill -0 "$(cat "$OUT/shard_${tag}.pid")" 2>/dev/null; do
    sleep 30
  done
  tail -1 "$OUT/shard_${tag}.log"
  touch "$OUT/shard_${tag}.done"
done

PYTHONPATH="$ROOT" "$PY" - <<'EOF'
import numpy as np
from pathlib import Path
out = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase7a/assets/p4t_dinov2")
parts = []
for tag in ["0-13016", "13016-26031", "26031-39047"]:
    z = np.load(out / f"shard_{tag}.npz")
    parts.append(z["feats"])
arr = np.concatenate(parts, axis=0)
np.savez_compressed(out / "feats.npz", feats=arr)
print("total", arr.shape)
EOF
echo "P4T_DINOV2_EXTRACT_DONE"
