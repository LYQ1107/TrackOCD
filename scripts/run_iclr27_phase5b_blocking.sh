#!/usr/bin/env bash
# TrackOCD ICLR 2027 — Phase 5B blocking runner (forensic audit).
set -euo pipefail
cd /data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT

PY=/home/lwr/anaconda3/envs/locatemot/bin/python
OUT=outputs/iclr27_phase5b
DEV_CSV=outputs/iclr27_phase4q/q1_long/proposals_dev.csv
FEATS=outputs/iclr27_phase4s/q1_features/feats.npz
PROTO=outputs/iclr27_phase5a/pilot/episodes

echo "== 00_preflight =="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h /data1
free -h

echo "== 02_freeze_artifacts =="
sha256sum outputs/iclr27_phase4q/q1_long/checkpoint.pth
stat -c '%s %y %n' outputs/iclr27_phase4q/q1_long/checkpoint.pth \
  outputs/iclr27_phase4q/q1_long/proposals_dev.csv \
  outputs/iclr27_phase4q/q1_long/teta_results/tao_track.json

echo "== 03_reproduce_phase5a_counts =="
$PY src/iclr27_phase5b/stream_audit/reproduce_counts.py \
  --csv $DEV_CSV --out $OUT/audit/counts

echo "== 04-07_trace_lineage_lifecycle =="
test -s docs/iclr27_phase5b/Q1_PHYSICAL_STREAM_DATA_LINEAGE.md
test -s docs/iclr27_phase5b/OVTR_LOCAL_DIFF_AUDIT.md

echo "== 13-17_geometry_alignment =="
$PY src/iclr27_phase5b/alignment/geometry_audit.py \
  --csv $DEV_CSV --out $OUT/audit/geometry

echo "== 18-20_tao_audit =="
$PY src/iclr27_phase5b/tao_audit/coverage.py --out $OUT/audit/tao_coverage
$PY src/iclr27_phase5b/tao_audit/annotation_status.py \
  --csv $DEV_CSV \
  --forensic $OUT/audit/geometry/track_forensic_table.csv \
  --out $OUT/audit/geometry/track_forensic_table_with_tao.csv

echo "== 26-27_retention_frontier =="
$PY src/iclr27_phase5b/retention_frontier.py --csv $DEV_CSV \
  --out $OUT/audit/retention

echo "== 21-22_visual_contact_sheets =="
$PY src/iclr27_phase5b/visualization/contact_sheets.py \
  --csv $DEV_CSV --n 300 --seed 20260816 --out $OUT/visual_audit

echo "== 24_counterfactual_streams =="
$PY src/iclr27_phase5b/counterfactual/replay.py \
  --csv $DEV_CSV --feats $FEATS \
  --forensic $OUT/audit/geometry/track_forensic_table.csv \
  --out $OUT/counterfactual

echo "== 25_frozen_semantic_replay =="
for s in S2_geom_aligned_oracle S3_dedup S4_frag_norm_diag; do
  $PY src/iclr27_phase5a/evaluation/strict_causal_eval.py \
    --proposals $OUT/counterfactual/${s}_proposals.csv \
    --feats $OUT/counterfactual/${s}_feats.npz \
    --proto-dir $PROTO --embed h --mode threshold --tau 0.75 \
    --ema-alpha 0.1 --update-threshold 0.85 --device cuda:0 \
    --out $OUT/counterfactual/replay_${s}
done

echo "== 28_prior_art =="
test -s docs/iclr27_phase5b/2025_2026_PHYSICAL_TRACK_LIFECYCLE_PRIOR_ART.md

echo "== 37_write_report =="
test -s docs/iclr27_phase5b/PHASE5B_COMPLETE_COPYABLE_REPORT.md
echo "Phase 5B blocking pipeline complete."
