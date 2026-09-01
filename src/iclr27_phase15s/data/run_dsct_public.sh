#!/usr/bin/env bash
# One blocking, bounded Phase15S public-TRAIN DSCT expansion.
set -euo pipefail
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
GPU=${PHASE15S_GPU:-2}
OUT=$ROOT/outputs/iclr27_phase15s/dsct_bank/public_roles
ANN=$ROOT/data/iclr27_phase15s/sources/validation_public_roles.json
CFG=$ROOT/src/iclr27_phase15s/data/phase15s_tao_train.py
CKPT=$ROOT/data/iclr27_phase15s/checkpoints/phase6b_dsct_stage_d.pth
mkdir -p "$OUT/teta_results" "$ROOT/outputs/iclr27_phase15s/logs"
echo "[phase15s] preflight" | tee "$ROOT/outputs/iclr27_phase15s/logs/dsct_public_preflight.log"
df -h /data1 | tee -a "$ROOT/outputs/iclr27_phase15s/logs/dsct_public_preflight.log"
free -h | tee -a "$ROOT/outputs/iclr27_phase15s/logs/dsct_public_preflight.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$ROOT/outputs/iclr27_phase15s/logs/dsct_public_preflight.log"
ps -eo stat= | wc -l | tee -a "$ROOT/outputs/iclr27_phase15s/logs/dsct_public_preflight.log"
if [[ -f "$OUT/.done" ]]; then echo "already complete"; exit 0; fi
if [[ -f "$OUT/.launched" ]]; then echo "unit already launched; refusing blind relaunch"; exit 2; fi
touch "$OUT/.launched"
VJSON=$(python - <<'PY'
import json
d=json.load(open('/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/iclr27_phase15s/manifests/data_split_and_leakage_audit.json'))
print(json.dumps(sorted(set(d['roles']['known_bank_train']) | set(d['roles']['known_calibration']) | set(d['roles']['known_audit'])), separators=(',', ':')))
PY
)
cd "$OVTR"
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PY" eval.py \
  --config_file "$CFG" --dataset_file lvis_generated_img_seqs --batch_size 1 \
  --with_box_refine --two_stage --pretrain "$CKPT" \
  --score_mode dsct --num_workers 2 --sampler_lengths 2 \
  --video_ids "$VJSON" \
  --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
  --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
  --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
  --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1 \
  --joint_stats_path "$OUT/joint_stats.json" --dscq_stats_path "$OUT/dscq_stats.json" \
  --output_dir "$OUT" --eval track --result_path_track "$OUT/teta_results" \
  > "$OUT/eval.log" 2>&1
cd "$ROOT"
test -s "$OUT/teta_results/tao_track.json"
python - "$OUT/teta_results/tao_track.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert isinstance(d,list) and d, 'empty/truncated DSCT output'
assert all({'image_id','bbox','score','video_id','track_id'} <= set(r) for r in d)
print('validated DSCT records',len(d))
PY
touch "$OUT/.done"
echo PHASE15S_DSCT_PUBLIC_DONE
