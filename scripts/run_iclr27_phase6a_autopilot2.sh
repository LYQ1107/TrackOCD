#!/usr/bin/env bash
# Phase 6A autopilot v2: wait for main training AND the (re)trained
# ablations; run filtered evals for all models; run the authoritative
# FULL_VAL eval for the final main model on GPU 0; rebuild the report.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase6a

echo "[autopilot2] waiting for main training..."
while kill -0 "$(cat "$OUT/training/main/train.pid")" 2>/dev/null; do
  sleep 60
done
echo "[autopilot2] main training finished"
if ! grep -q "Training time" "$OUT/training/main/train.log"; then
  echo "[autopilot2] WARNING: main training did not finish cleanly"
  touch "$OUT/training/main/MAIN_TRAIN_FAILED"
fi

echo "[autopilot2] starting FULL_VAL final main eval on GPU 0 (background)..."
cd "$ROOT"
if [[ ! -f "$OUT/q1/main_final/.done" ]]; then
  FULL_VAL=1 bash scripts/eval_phase6a_one.sh 0 \
    "$OUT/training/main/checkpoint.pth" main_final \
    > "$OUT/q1/main_final_autopilot.log" 2>&1 &
  FULLVAL_PID=$!
  echo $FULLVAL_PID > "$OUT/q1/main_final.pid"
else
  FULLVAL_PID=""
fi

echo "[autopilot2] waiting for ablation retraining..."
while [[ ! -f "$OUT/ablations/a5_no_dynamic_memory/.done" ]]; do
  sleep 60
done
echo "[autopilot2] ablations retrained"

echo "[autopilot2] filtered evals for all models..."
bash scripts/run_iclr27_phase6a_blocking.sh eval

if [[ -n "$FULLVAL_PID" ]]; then
  echo "[autopilot2] waiting for FULL_VAL main eval..."
  wait "$FULLVAL_PID"
  touch "$OUT/q1/main_final/.done"
fi

echo "[autopilot2] rebuilding complete report..."
python3 "$ROOT/src/iclr27_phase6a/report/build_complete_report.py"
echo "[autopilot2] DONE"
