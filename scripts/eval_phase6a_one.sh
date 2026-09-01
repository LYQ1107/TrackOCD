#!/usr/bin/env bash
# Evaluate one Phase 6A model on the frozen Q1 stream: OVTR joint eval ->
# proposals CSV -> strict-causal jointcsv -> physical eval -> causal tests.
set -euo pipefail

GPU=${1:?gpu}
CKPT=${2:?checkpoint}
NAME=${3:?name}
shift 3

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
OUT=$ROOT/outputs/iclr27_phase6a/q1/$NAME
OVTR_PY=/home/lwr/anaconda3/envs/ovtr/bin/python
CKPT=$(realpath "$CKPT")
Q1_VIDEOS="[88,90,122,291,334,888,931,1159,1232,1276,1572,1865,2254,2347,2564,2675,2690,2759,2802,2888]"

VIDEO_ARGS=()
if [[ "${FULL_VAL:-0}" != "1" ]]; then
  VIDEO_ARGS=(--video_ids "$Q1_VIDEOS")
fi

mkdir -p "$OUT/teta_results"
cd "$OVTR_DIR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$OVTR_PY" eval.py \
  --config_file ./config/ovtr_lite_train_val.py \
  --dataset_file lvis_generated_img_seqs --batch_size 1 \
  --with_box_refine --two_stage --pretrain "$CKPT" \
  --score_mode joint --num_workers 4 --sampler_lengths 2 \
  "${VIDEO_ARGS[@]}" \
  --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
  --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
  --joint_coef 1.0 --joint_alpha 0.1 --joint_state_dim 128 \
  --joint_stats_path "$OUT/joint_stats.json" \
  "$@" --output_dir "$OUT" --eval track \
  --result_path_track "$OUT/teta_results" > "$OUT/eval.log" 2>&1

cd "$ROOT"
PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
  --results-json "$OUT/teta_results/tao_track.json" \
  --out-prefix "$OUT/proposals" > "$OUT/convert.log" 2>&1

echo "[extract feats]"
PYTHONPATH="$ROOT" /home/lwr/anaconda3/envs/locatemot/bin/python \
  "$ROOT/src/iclr27_phase4s/features_q1.py" \
  --proposals "$OUT/proposals_dev.csv" \
  --out "outputs/iclr27_phase6a/q1/$NAME" \
  --device "cuda:$GPU" --batch 64 > "$OUT/feats.log" 2>&1

if [[ "${SKIP_STRICT:-0}" != "1" ]]; then
  NROWS=$(( $(wc -l < "$OUT/proposals_dev.csv") - 1 ))
  if (( NROWS <= 0 )); then
    mkdir -p "$ROOT/outputs/iclr27_phase6a/strict_eval/${NAME}_joint"
    cat > "$ROOT/outputs/iclr27_phase6a/strict_eval/${NAME}_joint/summary.json" <<EOF
{"strict":{},"legacy_first_frame":{},"legacy_last_frame":{},"n_rows":0,"n_records":0,"n_aligned_tracks":0}
EOF
    echo "EMPTY_STREAM_SKIP_STRICT" > "$OUT/strict.log"
  else
    PYTHONPATH="$ROOT" /home/lwr/anaconda3/envs/locatemot/bin/python \
      "$ROOT/src/iclr27_phase5a/evaluation/strict_causal_eval.py" \
      --proposals "$OUT/proposals_dev.csv" \
      --feats "outputs/iclr27_phase6a/q1/$NAME/feats.npz" \
      --proto-dir outputs/iclr27_phase5a/pilot/episodes \
      --embed h --mode jointcsv --filter aligned \
      --device "cuda:$GPU" \
      --out "$ROOT/outputs/iclr27_phase6a/strict_eval/${NAME}_joint" \
      > "$OUT/strict.log" 2>&1
  fi
else
  echo "SKIP_STRICT=1" > "$OUT/strict.log"
fi

PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase6a/evaluation/physical_eval.py" \
  --csv "$OUT/proposals_dev.csv" \
  --out "$ROOT/outputs/iclr27_phase6a/physical_eval/${NAME}.json" \
  > "$OUT/physical.log" 2>&1

PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase6a/tests/causal_contract_tests.py" \
  --csv "$OUT/proposals_dev.csv" \
  --out "$ROOT/outputs/iclr27_phase6a/strict_eval/${NAME}_causal_contract.json" \
  > "$OUT/contract.log" 2>&1

PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase6a/evaluation/objectness_audit.py" \
  --joint-stats "$OUT/joint_stats.json" \
  --out "$ROOT/outputs/iclr27_phase6a/strict_eval/${NAME}_objectness_audit.json" \
  > "$OUT/objectness_audit.log" 2>&1

echo "EVAL_DONE $NAME"
