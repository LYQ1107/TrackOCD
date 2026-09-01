#!/usr/bin/env bash
# Phase 4T hierarchical TrackOCD belief pipeline.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT"
export PYTHONPATH="$ROOT"
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
OUT=outputs/iclr27_phase4t

echo "[00] preflight"
free -h | head -2
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | head -4

echo "[05] convert real train stream"
$PY src/iclr27_phase4t/stream_data.py --track-json "$OUT/train_stream/teta/tao_track.json" --out-csv "$OUT/train_stream/proposals.csv"

echo "[06] DINO features for train stream"
CUDA_VISIBLE_DEVICES=2 $PY src/iclr27_phase4t/features_train.py \
  --proposals "$OUT/train_stream/proposals.csv" --out "$OUT/train_stream" --device cuda:0 --batch 96

echo "[07] train T1 (hierarchy + synthetic) on GPU1"
CUDA_VISIBLE_DEVICES=1 $PY src/iclr27_phase4t/train.py --out "$OUT/t1" \
  --data synthetic --use-hierarchy --use-defer --epochs 30 --episodes-per-epoch 128 --batch-size 4 --device cuda:0 &
T1=$!

echo "[08] train T2 (flat + real) on GPU3"
CUDA_VISIBLE_DEVICES=3 $PY src/iclr27_phase4t/train.py --out "$OUT/t2" \
  --data real --stream-csv "$OUT/train_stream/proposals.csv" --stream-feats "$OUT/train_stream/feats.npz" \
  --epochs 30 --episodes-per-epoch 128 --batch-size 4 --device cuda:0 &
T2=$!

echo "[09] train T3 (hierarchy + real, forced) on GPU4"
CUDA_VISIBLE_DEVICES=4 $PY src/iclr27_phase4t/train.py --out "$OUT/t3" \
  --data real --stream-csv "$OUT/train_stream/proposals.csv" --stream-feats "$OUT/train_stream/feats.npz" \
  --use-hierarchy --epochs 30 --episodes-per-epoch 128 --batch-size 4 --device cuda:0 &
T3=$!

echo "[10] train T4 (hierarchy + real + qphys + defer) on GPU5"
CUDA_VISIBLE_DEVICES=5 $PY src/iclr27_phase4t/train.py --out "$OUT/t4" \
  --data real --stream-csv "$OUT/train_stream/proposals.csv" --stream-feats "$OUT/train_stream/feats.npz" \
  --use-hierarchy --use-defer --use-qphys --epochs 30 --episodes-per-epoch 128 --batch-size 4 --device cuda:0 &
T4=$!
wait $T1 $T2 $T3 $T4

echo "[11] episodic pilots"
$PY src/iclr27_phase4t/pilot.py --checkpoint "$OUT/t1/checkpoint.pth" --data synthetic --use-hierarchy --use-defer --n-episodes 300 --out "$OUT/t1_pilot" --device cuda:1 &
$PY src/iclr27_phase4t/pilot.py --checkpoint "$OUT/t2/checkpoint.pth" --data real --n-episodes 300 --out "$OUT/t2_pilot" --device cuda:2 &
$PY src/iclr27_phase4t/pilot.py --checkpoint "$OUT/t3/checkpoint.pth" --data real --use-hierarchy --n-episodes 300 --out "$OUT/t3_pilot" --device cuda:3 &
$PY src/iclr27_phase4t/pilot.py --checkpoint "$OUT/t4/checkpoint.pth" --data real --use-hierarchy --use-defer --use-qphys --n-episodes 300 --out "$OUT/t4_pilot" --device cuda:4 &
wait

echo "[12] dev evals"
for m in t1 t2 t3 t4; do
  case $m in
    t1) flags="--use-hierarchy --use-defer";;
    t2) flags="";;
    t3) flags="--use-hierarchy";;
    t4) flags="--use-hierarchy --use-defer --use-qphys";;
  esac
  $PY src/iclr27_phase4t/dev_eval.py --checkpoint "$OUT/$m/checkpoint.pth" \
    --feats outputs/iclr27_phase4s/q1_features/feats.npz \
    $flags --out "$OUT/dev_$m" --device cuda:1 > "$OUT/dev_$m.out" 2>&1 &
done
wait
echo "PHASE4T_PIPELINE_DONE"
