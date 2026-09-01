#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4I blocking orchestrator (unique entry).
# Idempotent by artifact existence. Run:
#   bash scripts/run_iclr27_phase4i_blocking.sh 2>&1 |
#     tee runs/iclr27_phase4i/phase4i_blocking.log
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4i"
mkdir -p "$RUN_DIR"

pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F', ' '
      $2==0 && $3==0 {print $1; exit}'
}
have() { [ -f "$1" ]; }

echo "== 00 preflight =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv \
  | tee "$RUN_DIR/gpu_start.log"
free -h > "$RUN_DIR/mem_start.log"
df -h /data1 >> "$RUN_DIR/mem_start.log"
mkdir -p configs/iclr27_phase4i configs/frame_online_trackocd \
  docs/iclr27_phase4i docs/frame_online_trackocd \
  outputs/iclr27_phase4i outputs/frame_online_trackocd \
  runs/iclr27_phase4i runs/frame_online_trackocd \
  src/iclr27_phase4i src/frame_online_trackocd \
  tests/iclr27_phase4i tests/frame_online_trackocd \
  third_party/research_refs_phase4i

echo "== 03-06 input manifest + audits =="
$PY -u src/iclr27_phase4i/build_input_manifest.py > /dev/null
$PY -u src/iclr27_phase4i/association_manifest.py > /dev/null
for f in docs/iclr27_phase4i/CURRENT_TRACKER_ASSOCIATION_AUDIT.md \
         docs/iclr27_phase4i/FRAME_CAUSAL_DATAFLOW_AUDIT.md \
         docs/iclr27_phase4i/FRAME_ONLINE_TRACKOCD_PROTOCOL.md \
         docs/iclr27_phase4i/FRAME_CAUSAL_CONTRACT.md; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done

echo "== 09-12 open-source review =="
for f in repository_inventory association_mechanism_matrix; do
  [ -f "outputs/iclr27_phase4i/open_source/$f.csv" ] || \
    { echo "missing $f.csv"; exit 1; }
done

echo "== 07-08 B0 replay + equivalence =="
if ! have outputs/frame_online_trackocd/subset/b0_equivalence.json; then
  G=$(pick_gpu); [ -n "$G" ] || { echo "no free GPU"; exit 1; }
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4i/run_subset.py \
    --modes B0 --gpu "$G"
fi

echo "== 13-16 detection features + prefix audits =="
if ! have outputs/iclr27_phase4i/audit/prefix_semantic_stability.csv; then
  G=$(pick_gpu); [ -n "$G" ] || { echo "no free GPU"; exit 1; }
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4i/prefix_audit.py \
    --out-csv outputs/iclr27_phase4i/audit/prefix_semantic_stability.csv \
    --out-pairs-csv outputs/iclr27_phase4i/audit/prefix_positive_statistics.csv
fi

echo "== 17-21 B1/B2 subset runs + trackeval =="
for spec in "B1 _ 0.0" "B2 l0.1 0.1" "B2 l0.3 0.3"; do
  set -- $spec
  MODE=$1; TAG=$2; LS=${3:-0.25}
  PRED_DIR="outputs/frame_online_trackocd/subset/$MODE/${TAG:-_}"
  TE_OUT="outputs/frame_online_trackocd/subset/$MODE/trackeval.json"
  if [ "$MODE" = "B2" ]; then
    TE_OUT="outputs/frame_online_trackocd/subset/$MODE/$TAG/trackeval.json"
  fi
  if ! have "$TE_OUT"; then
    G=$(pick_gpu); [ -n "$G" ] || { echo "no free GPU"; exit 1; }
    CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4i/run_subset.py \
      --modes "$MODE" --lambda-s "$LS" --prefix-mode P1 --gpu "$G"
    $PY -u src/iclr27_phase4i/build_trackeval_input.py \
      --input-dir "$PRED_DIR" \
      --tracker-name "$MODE" \
      --output-root "${TE_OUT%/trackeval.json}/trackeval"
    $PY -u src/iclr27_phase4i/run_trackeval_subset.py \
      --trackers-folder "${TE_OUT%/trackeval.json}/trackeval" \
      --names "$MODE" \
      --out "$TE_OUT"
  fi
done

echo "== 22-24 failure / fp / fragment audits =="
[ -f outputs/iclr27_phase4i/audit/tracking_native_failure.csv ] || \
  $PY -u src/iclr27_phase4i/failure_audits.py \
    --pred-dir outputs/frame_online_trackocd/subset/B0/_ \
    --out-csv outputs/iclr27_phase4i/audit/tracking_native_failure.csv
[ -f outputs/iclr27_phase4i/audit/fragment_semantic_continuity.csv ] || \
  $PY -u src/iclr27_phase4i/fragment_continuity.py \
    --pred-dir outputs/frame_online_trackocd/subset/B0/_ \
    --out-csv outputs/iclr27_phase4i/audit/fragment_semantic_continuity.csv

echo "== 36-42 semantic eval + tests =="
for spec in "B1 " "B2 l0.1" "B2 l0.3"; do
  set -- $spec; MODE=$1; TAG=${2:-}
  LOGROOT="outputs/iclr27_phase4i/audit/semantic_logs"
  [ -d "$LOGROOT/$MODE${TAG:+_$TAG}" ] && \
    $PY -u src/iclr27_phase4i/semantic_eval.py \
      --log-root "$LOGROOT/$MODE${TAG:+_$TAG}" \
      --out "outputs/iclr27_phase4i/audit/semantic_eval_${MODE}${TAG:+_$TAG}.csv"
done
$PY -m pytest tests/iclr27_phase4i tests/frame_online_trackocd -q \
  --disable-warnings > "$RUN_DIR/pytest.log" 2>&1 || { tail -30 "$RUN_DIR/pytest.log"; exit 1; }
tail -3 "$RUN_DIR/pytest.log"

echo "== 43-45 reports =="
[ -f docs/iclr27_phase4i/PHASE4I_COMPLETE_COPYABLE_REPORT.md ] || \
  { echo "final report missing"; exit 1; }
echo "PHASE4I_BLOCKING_DONE"
