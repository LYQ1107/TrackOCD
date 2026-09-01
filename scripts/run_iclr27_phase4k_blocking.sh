#!/usr/bin/env bash
# TrackOCD ICLR 2027 Phase 4K blocking orchestrator (unique entry).
# Idempotent by artifact existence.  Run:
#   bash scripts/run_iclr27_phase4k_blocking.sh 2>&1 |
#     tee runs/iclr27_phase4k/phase4k_blocking.log
set -u
ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$ROOT" || exit 1
PY=/home/lwr/anaconda3/envs/locatemot/bin/python
export PYTHONPATH="$ROOT"
RUN_DIR="$ROOT/runs/iclr27_phase4k"
AUDIT="$ROOT/outputs/iclr27_phase4k/audit"
DOC="$ROOT/docs/iclr27_phase4k"
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
df -h /data1 /data3 >> "$RUN_DIR/mem_start.log"

echo "== 04-07 provenance replays =="
run_prov() {
  TAG=$1
  if have "$AUDIT/prov_$TAG/PHASE4K_PROVENANCE_DONE" &&
     [ "$(python -c "import json;print(json.load(open('$AUDIT/prov_$TAG/equivalence.json'))['byte_exact'])" 2>/dev/null)" = "True" ]; then
    echo "skip $TAG (done + byte-exact)"
    return
  fi
  G=$(pick_gpu); [ -n "$G" ] || G=$(pick_gpu_low)
  echo "run $TAG on GPU $G"
  CUDA_VISIBLE_DEVICES=$G $PY -u src/iclr27_phase4k/run_provenance.py \
    --tag "$TAG" --gpu "$G" > "$RUN_DIR/prov_$TAG.log" 2>&1 || {
    echo "FAILED $TAG"; tail -30 "$RUN_DIR/prov_$TAG.log"; exit 1; }
  python - "$TAG" <<'PY'
import json, pathlib, sys
tag = sys.argv[1]
eq = json.loads(pathlib.Path(
    f"outputs/iclr27_phase4k/audit/prov_{tag}/equivalence.json").read_text())
if not eq["byte_exact"]:
    print(f"NOT byte-exact {tag}: {eq}")
    sys.exit(1)
pathlib.Path(f"outputs/iclr27_phase4k/audit/prov_{tag}/"
             "PHASE4K_PROVENANCE_DONE").touch()
print(f"OK {tag} byte-exact")
PY
}
run_prov j0
run_prov j1b
run_prov m1

echo "== 08-13 offline audit + predictability + docs =="
for TAG in j0 j1b m1; do
  if ! have "$AUDIT/prototype_provenance_$TAG.csv"; then
    $PY -u src/iclr27_phase4k/build_offline_audit.py --tag "$TAG" || exit 1
  fi
done
if ! have "$AUDIT/causal_predictability.csv"; then
  for TAG in j1b j0 m1; do
    $PY -u src/iclr27_phase4k/causal_predictability.py --tag "$TAG" || exit 1
  done
fi
$PY -u src/iclr27_phase4k/generate_phase4k_docs.py || exit 1
cp "$AUDIT/prov_j1b/prototype_event_log_j1b.jsonl" "$AUDIT/" 2>/dev/null || true
cp "$AUDIT/prov_j1b/embeddings_j1b.npz" "$AUDIT/" 2>/dev/null || true

echo "== 14 root-cause decision =="
have "$DOC/ROOT_CAUSE_DECISION.md" || {
  echo "ROOT_CAUSE_DECISION.md missing; write it from the audit data"; exit 1; }

echo "== 15-18 open-source =="
for f in outputs/iclr27_phase4k/open_source/repository_inventory.csv \
         outputs/iclr27_phase4k/open_source/mechanism_matrix.csv \
         "$DOC/OPEN_SOURCE_REPOSITORY_AUDIT.md" \
         "$DOC/OPEN_SOURCE_IMPLEMENTATION_NOTES.md"; do
  have "$f" || { echo "missing $f"; exit 1; }
done

echo "== 19-28 method branch =="
grep -q PROGRESSIVE_MEMORY_NOT_SUPPORTED "$DOC/ROOT_CAUSE_DECISION.md" && {
  echo "method branch stopped by decision (no promotion development)"
}
grep -qE 'CAUSAL_PROMOTION_SIGNAL_(STRONG|PARTIAL)' \
  "$DOC/ROOT_CAUSE_DECISION.md" && {
  have "$DOC/CAUSAL_SEMANTIC_MEMORY_PROMOTION_SPEC.md" || {
    echo "promotion spec missing"; exit 1; }
  echo "method branch requires manual development + dev eval (see spec)"
}

echo "== 29-43 held-out =="
have "$DOC/HELDOUT_RESULTS.md" || {
  echo "HELDOUT_RESULTS.md missing"; exit 1; }
have "$DOC/GENERALIZATION_DECISION.md" || {
  echo "GENERALIZATION_DECISION.md missing"; exit 1; }

echo "== 44-47 novelty + tests =="
have "$DOC/PHASE4K_METHOD_NOVELTY_AUDIT.md" || {
  echo "PHASE4K_METHOD_NOVELTY_AUDIT.md missing"; exit 1; }
$PY -m pytest tests/iclr27_phase4k/test_phase4k_contracts.py -q \
  -p no:cacheprovider || exit 1

echo "== 48-50 integrated report =="
have "$DOC/PHASE4K_COMPLETE_COPYABLE_REPORT.md" || {
  echo "final report missing"; exit 1; }
echo "PHASE4K_BLOCKING_DONE"
