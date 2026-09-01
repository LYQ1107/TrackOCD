#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
OUT=$ROOT/outputs/iclr27_phase14c/smoke
CKPT=$ROOT/outputs/iclr27_phase6b/training/stage_d/checkpoint.pth
VIDS=${1:-'[0,51]'}

PYTHONPATH="$ROOT" python3 "$ROOT/src/iclr27_phase14c/data/build_smoke_annotation.py" \
  --videos "${SMOKE_VIDEO_IDS:-0,51}" \
  > "$ROOT/outputs/iclr27_phase14c/smoke/build_annotation.log"

mkdir -p "$OUT/teta_results"
cd "$OVTR_DIR"
CUDA_VISIBLE_DEVICES="${GPU:-0}" PYTHONPATH=. "$PY" eval.py \
  --config_file ./config/phase14c_smoke.py \
  --dataset_file lvis_generated_img_seqs --batch_size 1 \
  --with_box_refine --two_stage --pretrain "$CKPT" \
  --score_mode dsct --num_workers 1 --sampler_lengths 2 \
  --video_ids "$VIDS" \
  --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
  --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
  --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1 \
  --joint_stats_path "$OUT/joint_stats.json" \
  --dscq_stats_path "$OUT/dscq_stats.json" \
  --output_dir "$OUT" --eval track --result_path_track "$OUT/teta_results" \
  > "$OUT/eval.log" 2>&1
cd "$ROOT"
PYTHONPATH="$ROOT" python3 src/iclr27_phase14c/proposal/convert.py \
  --results-json "$OUT/teta_results/tao_track.json" \
  --annotation data/iclr27_phase14c/manifests/phase14c_smoke_validation.json \
  --out outputs/iclr27_phase14c/smoke/proposals.csv \
  > "$OUT/convert.log" 2>&1
PYTHONPATH="$ROOT" python3 - <<'PY'
import csv, json, pathlib, sys
p=pathlib.Path('outputs/iclr27_phase14c/smoke/proposals.csv')
rows=list(csv.DictReader(p.open()))
assert rows, 'smoke proposal stream is empty'
assert {'video_id','frame_id','track_id','bbox_xyxy','score','prior_hits'} <= set(rows[0])
assert len({int(r['video_id']) for r in rows}) >= 1
print(json.dumps({'rows':len(rows),'videos':len({int(r['video_id']) for r in rows}),'tracks':len({(int(r['video_id']),int(r['track_id'])) for r in rows})}))
PY
touch "$OUT/.done"
echo "PHASE14C_SMOKE_DONE"
