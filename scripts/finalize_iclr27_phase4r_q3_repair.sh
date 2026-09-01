#!/usr/bin/env bash
# Blocking finalizer for the repaired Q3 pilot eval (pid in eval_repair.pid).
# Waits once, validates the TETA summary, converts to the frozen proposal
# protocol, and merges the Q0/Q1/Q2/Q2-alpha/Q3 pre-repair/Q3 repair table.
set -u

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase4r
PY=python3
Q4Q=$ROOT/outputs/iclr27_phase4q

pid=$(pgrep -f "eval.py.*iclr27_phase4r/q3_pilot" | head -1 || true)
if [ -z "$pid" ]; then
  pid=$(cat "$OUT/q3_pilot/eval_repair.pid" 2>/dev/null || true)
fi
echo "[finalize] waiting for eval pid $pid"
start=$(date +%s)
last_print=$start
while ! grep -q "Saved the TETA summary results" "$OUT/q3_pilot/eval_repair.log" 2>/dev/null; do
  if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
    echo "[finalize] eval process exited without TETA marker"
    break
  fi
  now=$(date +%s)
  if [ $((now - last_print)) -ge 1500 ]; then
    last_print=$now
    el=$(((now - start) / 60))
    prog=$(tail -c 4000 "$OUT/q3_pilot/eval_repair.log" | tr '\r' '\n' \
      | grep -oE '[0-9]+/36375 \[[0-9:]+<[^]]*\]' | tail -n 1)
    echo "[finalize] ${el} min elapsed: ${prog:-starting}"
  fi
  sleep 60
done
echo "[finalize] eval finished"

if ! grep -q "Saved the TETA summary results" "$OUT/q3_pilot/eval_repair.log"; then
  echo "FINALIZE_FAILED: no TETA summary marker"
  tail -n 20 "$OUT/q3_pilot/eval_repair.log" | tr '\r' '\n' | tail -n 12
  exit 1
fi
if [ ! -s "$OUT/q3_pilot/teta_results/tao_track.json" ]; then
  echo "FINALIZE_FAILED: missing tao_track.json"
  exit 1
fi

echo "[finalize] converting repaired Q3 proposals"
PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
  --results-json "$OUT/q3_pilot/teta_results/tao_track.json" \
  --out-prefix "$OUT/q3_pilot/proposals"

"$PY" "$ROOT/src/iclr27_phase4r/finalize_q3_eval.py" \
  --out "$OUT/audits/q3_eval_compare.json" \
  --model "q0=$Q4Q/q0_long/eval.log=$Q4Q/q0_long/proposals" \
  --model "q1=$Q4Q/q1_long/eval.log=$Q4Q/q1_long/proposals" \
  --model "q2=$Q4Q/q2_long/eval.log=$Q4Q/q2_long/proposals" \
  --model "q2_a010=$OUT/q2_alpha/a010/eval.log=$OUT/q2_alpha/a010/proposals" \
  --model "q2_a025=$OUT/q2_alpha/a025/eval.log=$OUT/q2_alpha/a025/proposals" \
  --model "q3_pre=$OUT/q3_pilot/eval.log=$OUT/audits/q3_pilot_pre_repair" \
  --model "q3_repair=$OUT/q3_pilot/eval_repair.log=$OUT/q3_pilot/proposals"
