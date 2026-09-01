#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4L blocking orchestrator (unique entry).
# Idempotent by artifact existence.
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4l"
AUDIT="$ROOT/outputs/iclr27_phase4l/audit"
DOC="$ROOT/docs/iclr27_phase4l"
mkdir -p "$RUN_DIR" "$AUDIT" "$DOC"

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
df -h /data1 /data2 /data3 >> "$RUN_DIR/mem_start.log"

echo "== 04 anchor reproduction (Phase 4K frozen J1b) =="
python - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
eq = json.loads((root / "outputs/iclr27_phase4k/audit/prov_j1b/"
                  "equivalence.json").read_text())
assert eq["byte_exact"] is True
print("J1b anchor byte-exact: OK")
PY

echo "== 05-09 admissibility audit =="
if ! have "$AUDIT/admissibility_detection_features.csv" || \
   ! have "$AUDIT/admissibility_tracklet_features.csv"; then
  G=$(pick_gpu); [ -n "$G" ] || G=$(pick_gpu_low)
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4l/build_admissibility_dataset.py \
    || exit 1
fi
if ! have "$AUDIT/admissibility_predictability.csv"; then
  $PY -u src/iclr27_phase4l/admissibility_predictability.py || exit 1
fi

echo "== 11-17 novel matching audit =="
if ! have "$AUDIT/novel_matching_pairs.csv"; then
  G=$(pick_gpu); [ -n "$G" ] || G=$(pick_gpu_low)
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4l/build_novel_matching_dataset.py \
    || exit 1
fi
if ! have "$AUDIT/relative_matching_predictability.csv"; then
  $PY -u src/iclr27_phase4l/relative_matching_predictability.py || exit 1
fi

echo "== audit docs =="
$PY -u src/iclr27_phase4l/generate_phase4l_docs.py || exit 1

echo "== 19-22 open-source =="
for f in outputs/iclr27_phase4l/open_source/repository_inventory.csv \
         outputs/iclr27_phase4l/open_source/mechanism_matrix.csv \
         "$DOC/OPEN_SOURCE_REPOSITORY_AUDIT.md" \
         "$DOC/OPEN_SOURCE_IMPLEMENTATION_NOTES.md"; do
  have "$f" || { echo "missing $f"; exit 1; }
done

echo "== 10+18 root decisions =="
have "$DOC/ADMISSIBILITY_ROOT_DECISION.md" || {
  echo "ADMISSIBILITY_ROOT_DECISION.md missing"; exit 1; }
have "$DOC/MATCHING_ROOT_DECISION.md" || {
  echo "MATCHING_ROOT_DECISION.md missing"; exit 1; }

echo "== 23-30 conditional method branch (manual / candidate scripts) =="
grep -qE 'ADMISSIBILITY_SIGNAL_(STRONG|PARTIAL)' \
  "$DOC/ADMISSIBILITY_ROOT_DECISION.md" || \
  echo "admissibility branch not supported (skip)"
grep -qE 'RELATIVE_MATCHING_SIGNAL_(STRONG|PARTIAL)' \
  "$DOC/MATCHING_ROOT_DECISION.md" || \
  echo "matching branch not supported (skip)"

echo "== 31-37 held-out =="
have "$ROOT/outputs/iclr27_phase4l/heldout/selected_heldout_videos.csv" || {
  echo "held-out selection missing"; exit 1; }
have "$ROOT/outputs/iclr27_phase4l/heldout/validation_heldout_tao.json" || {
  echo "held-out GT missing"; exit 1; }
have "$DOC/HELDOUT_RESULTS.md" || {
  echo "HELDOUT_RESULTS.md missing"; exit 1; }

echo "== 46-50 novelty + tests + report =="
have "$DOC/PHASE4L_METHOD_NOVELTY_AUDIT.md" || {
  echo "PHASE4L_METHOD_NOVELTY_AUDIT.md missing"; exit 1; }
have "$DOC/PHASE4L_COMPLETE_COPYABLE_REPORT.md" || {
  echo "final report missing"; exit 1; }
$PY -m pytest tests/iclr27_phase4l/test_phase4l_contracts.py -q \
  -p no:cacheprovider || exit 1
echo "PHASE4L_BLOCKING_DONE"
