#!/usr/bin/env bash
# Blocking finalizer for the three Phase 4R evals already in flight:
#   q3_pilot (pid 37398), q2_alpha/a010 (18837), q2_alpha/a025 (18838).
# Waits with a single supervisor loop, validates the TETA summary, converts
# TAO JSON to the frozen proposal protocol, and merges Q0/Q1/Q2/Q2-alpha/Q3.
set -u

ROOT=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
OUT=$ROOT/outputs/iclr27_phase4r
PY=python3
Q4Q=$ROOT/outputs/iclr27_phase4q

PIDS=(37398 18837 18838)
DIRS=(q3_pilot q2_alpha/a010 q2_alpha/a025)

echo "[finalize] waiting for eval PIDs: ${PIDS[*]}"
start=$(date +%s)
last_print=$start
while true; do
  alive=""
  for p in "${PIDS[@]}"; do
    kill -0 "$p" 2>/dev/null && alive="$alive $p"
  done
  if [ -z "$alive" ]; then
    echo "[finalize] all eval processes exited"
    break
  fi
  now=$(date +%s)
  if [ $((now - last_print)) -ge 1500 ]; then
    last_print=$now
    el=$(((now - start) / 60))
    echo "[finalize] ${el} min elapsed; alive:${alive}"
    for d in "${DIRS[@]}"; do
      log=$OUT/$d/eval.log
      if [ -f "$log" ]; then
        prog=$(tail -c 4000 "$log" | tr '\r' '\n' \
          | grep -oE '[0-9]+/36375 \[[0-9:]+<[^]]*\]' | tail -n 1)
        echo "    $d: ${prog:-starting}"
      fi
    done
  fi
  sleep 60
done

fail=0
for d in "${DIRS[@]}"; do
  dir=$OUT/$d
  json=$dir/teta_results/tao_track.json
  if [ ! -s "$json" ]; then
    echo "[finalize] FAIL: missing $json"
    fail=1
    continue
  fi
  if ! grep -q "Saved the TETA summary results" "$dir/eval.log"; then
    echo "[finalize] FAIL: $d eval.log has no TETA summary marker"
    tail -n 12 "$dir/eval.log" | tr '\r' '\n' | tail -n 8
    fail=1
    continue
  fi
  echo "[finalize] $d TETA summary present; converting proposals"
  PYTHONPATH="$ROOT" "$PY" "$ROOT/src/iclr27_phase4p/ovtr_main_eval.py" \
    --results-json "$json" \
    --out-prefix "$dir/proposals" || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "FINALIZE_FAILED"
  exit 1
fi

"$PY" "$ROOT/src/iclr27_phase4r/finalize_q3_eval.py" --out "$OUT/audits/q3_eval_compare.json" \
  --model "q0=$Q4Q/q0_long/eval.log=$Q4Q/q0_long/proposals" \
  --model "q1=$Q4Q/q1_long/eval.log=$Q4Q/q1_long/proposals" \
  --model "q2=$Q4Q/q2_long/eval.log=$Q4Q/q2_long/proposals" \
  --model "q2_a010=$OUT/q2_alpha/a010/eval.log=$OUT/q2_alpha/a010/proposals" \
  --model "q2_a025=$OUT/q2_alpha/a025/eval.log=$OUT/q2_alpha/a025/proposals" \
  --model "q3_pilot=$OUT/q3_pilot/eval.log=$OUT/q3_pilot/proposals"
