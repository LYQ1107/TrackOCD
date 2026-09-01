#!/usr/bin/env bash
# Evaluate all Phase 6B ablations on the frozen Q1 20-video stream
# (filtered protocol, strict skipped), one per GPU, in parallel.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase6b
Q1_VIDEOS="[88,90,122,291,334,888,931,1159,1232,1276,1572,1865,2254,2347,2564,2675,2690,2759,2802,2888]"

eval_one() {
  local name=$1 gpu=$2
  local dir=$OUT/ablations/$name
  [[ -f "$dir/eval.done" ]] && { echo "$name eval done"; return; }
  mkdir -p "$OUT/q1/abl_$name"
  SKIP_STRICT=1 bash "$ROOT/scripts/eval_phase6b_one.sh" "$gpu" \
    "$dir/checkpoint.pth" "abl_$name" --video_ids "$Q1_VIDEOS" \
    > "$OUT/q1/abl_$name/run.log" 2>&1
  touch "$dir/eval.done"
}

eval_one a2_no_p2s 7
eval_one a4_no_struct 7
eval_one a3_no_s2p 9
eval_one a5_knownconf 9
echo "ABLATION_EVALS_DONE"
