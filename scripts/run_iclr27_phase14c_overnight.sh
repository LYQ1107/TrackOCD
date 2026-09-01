#!/usr/bin/env bash
# One blocking, resumable primary DSCT proposal run for Phase14C.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OVTR_DIR=$ROOT/third_party/research_refs_phase4n/OVTR/ovtr
PY=/home/lwr/anaconda3/envs/ovtr/bin/python
OUT=$ROOT/outputs/iclr27_phase14c
CKPT=$ROOT/outputs/iclr27_phase6b/training/stage_d/checkpoint.pth
CONFIG=./config/phase14c_tao_train.py
GPUS=${PHASE14C_GPUS:-0}
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM=${#GPU_LIST[@]}; (( NUM > 0 && NUM <= 4 ))

mkdir -p "$OUT"/{proposals/shards,logs,markers}
echo "[$(date -Is)] preflight" | tee "$OUT/logs/overnight_preflight.log"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv | tee -a "$OUT/logs/overnight_preflight.log"
free -h | tee -a "$OUT/logs/overnight_preflight.log"
ps -eo stat= | wc -l | tee -a "$OUT/logs/overnight_preflight.log"
df -h /data1 | tee -a "$OUT/logs/overnight_preflight.log"

PYTHONPATH="$ROOT" python3 src/iclr27_phase14c/data/make_shards.py \
  --num-shards "$NUM" --out outputs/iclr27_phase14c/manifests/shards.json \
  > "$OUT/logs/shards.log"

is_alive() { [[ -f "$1" ]] && kill -0 "$(cat "$1")" 2>/dev/null; }
run_shard() {
  local idx=$1 gpu=$2 vids=$3 dir="$OUT/proposals/shards/shard_$1"
  mkdir -p "$dir/teta_results"
  if [[ -f "$dir/.done" ]]; then echo "shard $idx already complete"; return 0; fi
  if is_alive "$dir/pid"; then echo "shard $idx already running pid=$(cat "$dir/pid")"; return 0; fi
  touch "$dir/.launched"
  cd "$OVTR_DIR"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$PY" eval.py \
    --config_file "$CONFIG" --dataset_file lvis_generated_img_seqs --batch_size 1 \
    --with_box_refine --two_stage --pretrain "$CKPT" \
    --score_mode dsct --num_workers 2 --sampler_lengths 2 \
    --video_ids "$vids" \
    --score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --filter_score_thresh 0.19 0.19 0.19 0.19 0.19 0.19 0.19 \
    --ious_thresh 0.45 0.45 0.45 0.45 0.45 0.45 0.45 \
    --miss_tolerance 5 5 5 5 5 5 5 --maximum_quantity 160 \
    --dsct_coef 1.0 --dsct_state_dim 128 --dsct_alpha 0.1 \
    --joint_stats_path "$dir/joint_stats.json" --dscq_stats_path "$dir/dscq_stats.json" \
    --output_dir "$dir" --eval track --result_path_track "$dir/teta_results" \
    > "$dir/eval.log" 2> "$dir/eval.err" &
  echo $! > "$dir/pid"
  echo "launched shard $idx gpu=$gpu pid=$! videos=$vids"
  cd "$ROOT"
}

mapfile -t SHARDS < <(python3 - <<'PY'
import json
d=json.load(open('outputs/iclr27_phase14c/manifests/shards.json'))
for s in d['shards']:
 print(json.dumps(s['video_ids'], separators=(',',':')))
PY
)
for i in "${!SHARDS[@]}"; do run_shard "$i" "${GPU_LIST[$i]}" "${SHARDS[$i]}"; done

# One blocking wait for all workers; no agent-level polling is needed.
for i in "${!SHARDS[@]}"; do
  dir="$OUT/proposals/shards/shard_$i"
  if [[ -f "$dir/pid" ]]; then wait "$(cat "$dir/pid")" || { echo "shard $i failed" >&2; exit 1; }; fi
  test -s "$dir/teta_results/tao_track.json"
  python3 - "$dir/teta_results/tao_track.json" "$dir" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); assert isinstance(d,list) and d, 'empty/truncated OVTR JSON'
for r in d:
 assert {'image_id','bbox','score','video_id','track_id'} <= set(r), r
assert all(len(r['bbox']) >= 4 for r in d)
pathlib.Path(sys.argv[2], '.json_ok').touch()
PY
  PYTHONPATH="$ROOT" python3 src/iclr27_phase14c/proposal/convert.py \
    --results-json "$dir/teta_results/tao_track.json" \
    --out "outputs/iclr27_phase14c/proposals/shards/shard_$i.csv" \
    > "$dir/convert.log" 2>&1
  test -s "$OUT/proposals/shards/shard_$i.csv"
  touch "$dir/.done"
done
echo "PHASE14C_OVERNIGHT_DONE"
