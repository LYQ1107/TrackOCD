#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/ovtr/bin/python"; TAG="${1:-formal_r1}"; OUT="$ROOT/outputs/iclr27_phase85"; mkdir -p "$OUT/metrics" "$OUT/completion"; declare -a pids
for f in 0 1 2; do "$PY" "$ROOT/scripts/iclr27_phase85/train_physical_gate.py" --fold "$f" --tag "$TAG" --epochs 15 > "$OUT/metrics/physical_gate_${TAG}_f${f}.stdout" 2>&1 & pids[$f]=$!; done
status=0; for f in 0 1 2; do if ! wait "${pids[$f]}"; then status=1; fi; done; exit "$status"
