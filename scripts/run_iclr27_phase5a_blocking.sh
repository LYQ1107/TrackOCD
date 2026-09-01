#!/usr/bin/env bash
# TrackOCD ICLR 2027 — Phase 5A blocking runner (reproducibility).
# Every step is a single blocking command; no agent-level polling.
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT

PY=python
DEV=${DEV:-cuda:0}
OUT=outputs/iclr27_phase5a
EP=$OUT/pilot/episodes

echo "== 00_preflight =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /data1
free -h

echo "== 05_build_strict_oracle_ceiling =="
$PY src/iclr27_phase5a/protocol/strict_oracle.py --device $DEV \
  --out $OUT/protocol_audit/strict_oracle_q1

echo "== 25_metadev_pilot_episodes =="
$PY src/iclr27_phase5a/episodes/build_pilot_episodes.py \
  --n-train 150 --n-metadev 80 --seed 20260816 --max-len 12 \
  --known-set-sizes 4,6,12,24 --metadev-known-set-sizes 4,6 \
  --fp-per-episode 4 --device $DEV --out $EP

echo "== 26_pilot_gates =="
$PY src/iclr27_phase5a/pilot_gates.py --data $EP --out $OUT/pilot/gates \
  --tau-grid 0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80 \
  --ema-alpha 0.5
$PY src/iclr27_phase5a/alpha_sweep.py --data $EP --out $OUT/pilot/gates \
  --taus 0.70,0.75,0.80 --alphas 0.1,0.2 \
  --update-thresholds none,0.80,0.85

echo "== 18_learned_creation_head =="
$PY src/iclr27_phase5a/training/train_creation_head.py \
  --data $EP --out $OUT/pilot/head --epochs 40 --lr 1e-3 \
  --seed 20260816 --ema-alpha 0.1 --batch-size 256

echo "== 19-24_unit_tests =="
$PY src/iclr27_phase5a/evaluation/unit_tests.py

echo "== 30-31_q1_evaluations =="
$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed h --mode threshold --tau 0.75 \
  --ema-alpha 0.1 --update-threshold 0.85 --device $DEV \
  --out $OUT/strict_causal_eval/q1_threshold

$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed h --mode threshold --tau 0.75 \
  --ema-alpha 0.1 --update-threshold 0.85 --filter aligned --device $DEV \
  --out $OUT/strict_causal_eval/q1_threshold_aligned_only

$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed h --mode threshold --tau 0.75 \
  --ema-alpha 0.1 --update-threshold 0.85 --filter aligned \
  --no-update-novel --device $DEV \
  --out $OUT/strict_causal_eval/q1_threshold_aligned_only_static

$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed f --mode threshold --tau 0.80 \
  --filter aligned --device $DEV \
  --out $OUT/strict_causal_eval/q1_frame_aligned_only

$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed h --mode head \
  --head-checkpoint $OUT/pilot/head/head.pth --ema-alpha 0.1 \
  --filter aligned --device $DEV \
  --out $OUT/strict_causal_eval/q1_head_aligned_only

echo "== 27_one_major_repair_birth_gate =="
$PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
  --proposals outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  --feats outputs/iclr27_phase4s/q1_features/feats.npz \
  --proto-dir $EP --embed h --mode threshold --tau 0.75 \
  --ema-alpha 0.1 --update-threshold 0.85 --filter all \
  --min-birth-age 2 --min-birth-score 0.35 --min-birth-prior 1 \
  --device $DEV --out $OUT/strict_causal_eval/q1_threshold_birthgate

echo "== 49_multiseed =="
$PY src/iclr27_phase5a/episodes/build_pilot_episodes.py \
  --n-train 150 --n-metadev 80 --seed 20260817 --max-len 12 \
  --known-set-sizes 4,6,12,24 --metadev-known-set-sizes 4,6 \
  --fp-per-episode 4 --device $DEV --out $EP/seed_20260817
$PY src/iclr27_phase5a/episodes/build_pilot_episodes.py \
  --n-train 150 --n-metadev 80 --seed 20260818 --max-len 12 \
  --known-set-sizes 4,6,12,24 --metadev-known-set-sizes 4,6 \
  --fp-per-episode 4 --device $DEV --out $EP/seed_20260818
$PY src/iclr27_phase5a/multiseed_pilot.py --base $EP \
  --seeds base,20260817,20260818 --out $OUT/multiseed \
  --tau 0.75 --ema-alpha 0.1 --update-threshold 0.85

echo "== 53_final_report =="
test -s docs/iclr27_phase5a/PHASE5A_COMPLETE_COPYABLE_REPORT.md
echo "Phase 5A blocking pipeline complete."
