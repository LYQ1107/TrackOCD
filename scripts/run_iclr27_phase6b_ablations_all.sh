#!/usr/bin/env bash
# Launch the Phase 6B ablation driver on GPU 7 and GPU 9 in parallel and
# wait for both; exposes a single pid for the blocking runner.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase6b

setsid nohup env ABLATION_GPU=7 \
  ABLATION_LIST="a2_no_p2s a4_no_struct" \
  bash "$ROOT/scripts/run_iclr27_phase6b_ablations.sh" \
  > "$OUT/ablations/driver7.log" 2>&1 &
P7=$!
setsid nohup env ABLATION_GPU=9 \
  ABLATION_LIST="a3_no_s2p a5_knownconf" \
  bash "$ROOT/scripts/run_iclr27_phase6b_ablations.sh" \
  > "$OUT/ablations/driver9.log" 2>&1 &
P9=$!
echo "ablations drivers: gpu7=$P7 gpu9=$P9"
wait "$P7" "$P9"
echo "ABLATIONS_ALL_DONE"
