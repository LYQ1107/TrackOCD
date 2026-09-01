#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-policy_formal}"; STEPS="${2:-1000}"; ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/locatemot/bin/python"; mkdir -p "$ROOT/outputs/iclr27_phase43/logs" "$ROOT/outputs/iclr27_phase43/completion"; pids=()
for f in 0 1 2 3; do if [[ -f "$ROOT/outputs/iclr27_phase43/completion/${TAG}_f${f}.done" || -f "$ROOT/outputs/iclr27_phase43/completion/${TAG}_f${f}.launched" ]]; then continue; fi; gpu=$((f+4)); CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/iclr27_phase43/train_policy_gate.py" --fold "$f" --steps "$STEPS" --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu" >"$ROOT/outputs/iclr27_phase43/logs/${TAG}_f${f}.log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; exit "$status"
