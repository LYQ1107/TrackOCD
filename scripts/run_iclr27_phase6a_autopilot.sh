#!/usr/bin/env bash
# Phase 6A autopilot: wait for ablations, run all model evals, wait for the
# main training run, run the final main eval, and rebuild the report.
set -euo pipefail

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase6a

echo "[autopilot] waiting for ablation driver..."
while [[ ! -f "$OUT/ablations/a5_no_dynamic_memory/.done" ]]; do
  sleep 60
done
echo "[autopilot] ablations done"

cd "$ROOT"
bash scripts/run_iclr27_phase6a_blocking.sh eval

echo "[autopilot] waiting for main training to finish..."
while kill -0 "$(cat "$OUT/training/main/train.pid")" 2>/dev/null; do
  sleep 60
done
echo "[autopilot] main training finished"
if ! grep -q "Training time" "$OUT/training/main/train.log"; then
  echo "[autopilot] WARNING: main training did not finish cleanly"
  touch "$OUT/training/main/MAIN_TRAIN_FAILED"
fi

echo "[autopilot] final main eval..."
if [[ ! -f "$OUT/q1/main_final/.done" ]]; then
  FULL_VAL=1 bash scripts/eval_phase6a_one.sh 9 \
    "$OUT/training/main/checkpoint.pth" main_final
  touch "$OUT/q1/main_final/.done"
fi

echo "[autopilot] rebuilding complete report..."
python3 "$ROOT/src/iclr27_phase6a/report/build_complete_report.py"

echo "[autopilot] DONE"
