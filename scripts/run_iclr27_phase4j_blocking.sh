#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4J blocking orchestrator (unique entry).
# Idempotent by artifact existence.  Run:
#   bash scripts/run_iclr27_phase4j_blocking.sh 2>&1 |
#     tee runs/iclr27_phase4j/phase4j_blocking.log
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
PY_TRACKEVAL=/home/lwr/anaconda3/envs/AVI/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4j"
mkdir -p "$RUN_DIR"

have() { [ -f "$1" ]; }

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
      $2==0 && $3==0 {print $1; exit}'
}
pick_gpu_low() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
      {print $2, $3, $1}' | sort -n | head -1 | awk '{print $3}'
}

echo "== 00 preflight =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  | tee "$RUN_DIR/gpu_start.log"
free -h > "$RUN_DIR/mem_start.log"
df -h /data1 /data3 >> "$RUN_DIR/mem_start.log"

echo "== 03-06 audits =="
for f in outputs/iclr27_phase4j/audit/semantic_by_track_age.csv \
         outputs/iclr27_phase4j/audit/fp_lifetime.csv \
         outputs/iclr27_phase4j/audit/prefix_routing_by_age.csv; do
  have "$f" || { echo "missing $f"; exit 1; }
done

echo "== 07-10 open-source =="
for f in outputs/iclr27_phase4j/open_source/repository_inventory.csv \
         outputs/iclr27_phase4j/open_source/mechanism_matrix.csv; do
  have "$f" || { echo "missing $f"; exit 1; }
done

echo "== 11 train-side calibration =="
if ! have outputs/iclr27_phase4j/calibration/calibration_config.json; then
  G=$(pick_gpu); [ -n "$G" ] || G=$(pick_gpu_low)
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4j/train_side_gate_calibration.py \
    --device cuda || exit 1
fi

echo "== 12-15 subset runs (J0/J1/J1b/J2/J2b/J2_age3/J2_c0) =="
run_tag() {
  TAG=$1; shift
  if ! have "outputs/iclr27_phase4j/subset/$TAG/PHASE4J_SUBSET_DONE"; then
    G=$(pick_gpu); [ -n "$G" ] || G=$(pick_gpu_low)
    CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4j/run_subset.py \
      --tag "$TAG" --gpu "$G" "$@"
    touch "outputs/iclr27_phase4j/subset/$TAG/PHASE4J_SUBSET_DONE"
  fi
}
run_tag J0 --decision-threshold 0.5 --commit-mode M0
run_tag J1 --decision-threshold 0.15 --commit-mode M0
run_tag J1b --decision-threshold 0.3 --commit-mode M0
run_tag J2 --decision-threshold 0.15 --commit-mode M1 \
  --commit-min-age 2 --commit-min-support 2
run_tag J2b --decision-threshold 0.3 --commit-mode M1 \
  --commit-min-age 2 --commit-min-support 2
run_tag J2_age3 --decision-threshold 0.15 --commit-mode M1 \
  --commit-min-age 3 --commit-min-support 3
run_tag J2_c0 --decision-threshold 0.5 --commit-mode M1 \
  --commit-min-age 2 --commit-min-support 2

echo "== J0 equivalence vs Phase 4I B2 lambda=0.1 =="
$PY -u src/iclr27_phase4j/check_j0_equivalence.py

echo "== trackeval (AVI env, report convention) =="
for TAG in J0 J1 J1b J2 J2b J2_age3 J2_c0; do
  PRED="outputs/iclr27_phase4j/subset/$TAG"
  TE="outputs/iclr27_phase4j/trackeval/$TAG"
  if ! have "$TE/trackeval_flat.json"; then
    $PY -u src/iclr27_phase4i/build_trackeval_input.py \
      --input-dir "$PRED" --tracker-name "$TAG" \
      --output-root "$TE/trackeval"
    $PY_TRACKEVAL -u src/iclr27_phase4i/run_trackeval_subset.py \
      --trackers-folder "$TE/trackeval" --names "$TAG" \
      --out "$TE/trackeval.json"
    $PY -u src/iclr27_phase4j/trackeval_metrics.py \
      --trackeval-json "$TE/trackeval.json" --out "$TE/trackeval_flat.json"
  fi
done

echo "== semantic eval + fragment continuity =="
for TAG in J0 J1 J1b J2 J2b J2_age3 J2_c0; do
  LOG="outputs/iclr27_phase4j/semantic_logs/$TAG"
  PRED="outputs/iclr27_phase4j/subset/$TAG"
  if ! have "outputs/iclr27_phase4j/audit/semantic_eval_$TAG.csv"; then
    $PY -u src/iclr27_phase4j/semantic_eval.py \
      --log-root "$LOG" \
      --out "outputs/iclr27_phase4j/audit/semantic_eval_$TAG.csv" \
      --out-tracklets "outputs/iclr27_phase4j/audit/tracklets_$TAG.csv" \
      --by-age-csv "outputs/iclr27_phase4j/audit/by_age_$TAG.csv"
  fi
  if ! have "outputs/iclr27_phase4j/audit/fragment_$TAG.csv"; then
    $PY -u src/iclr27_phase4j/fragment_continuity.py \
      --pred-dir "$PRED" --sem-log-root "$LOG" \
      --out-csv "outputs/iclr27_phase4j/audit/fragment_$TAG.csv"
  fi
done

echo "== 20-23 compare + error transfer =="
if ! have outputs/iclr27_phase4j/compare_candidates.csv; then
  $PY -u src/iclr27_phase4j/compare_candidates.py \
    --tags J0 J1 J1b J2 J2b J2_age3 J2_c0
fi

echo "== 39 tests =="
$PY -m pytest tests/iclr27_phase4j tests/frame_online_trackocd \
  tests/iclr27_phase4i -q --disable-warnings > "$RUN_DIR/pytest.log" 2>&1 \
  || { tail -30 "$RUN_DIR/pytest.log"; exit 1; }
tail -3 "$RUN_DIR/pytest.log"

echo "== 40-42 reports =="
have docs/iclr27_phase4j/PHASE4J_COMPLETE_COPYABLE_REPORT.md || \
  { echo "final report missing"; exit 1; }
echo "PHASE4J_BLOCKING_DONE"
