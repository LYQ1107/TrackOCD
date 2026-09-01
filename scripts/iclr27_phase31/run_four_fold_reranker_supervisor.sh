#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="/home/lwr/anaconda3/envs/locatemot/bin/python"; TAG="${1:-rawrerank_formal}"; STEPS="${STEPS:-2000}"
mkdir -p "$ROOT/outputs/iclr27_phase31/logs" "$ROOT/outputs/iclr27_phase31/completion"; pids=()
for fold in 0 1 2 3; do gpu=$((fold+4)); donef="$ROOT/outputs/iclr27_phase31/completion/${TAG}_f${fold}.done"; launch="$ROOT/outputs/iclr27_phase31/completion/${TAG}_f${fold}.launched"; if [[ -e "$donef" ]]; then continue; fi; [[ ! -e "$launch" ]] || { echo "launched marker exists for fold $fold" >&2; exit 2; }; log="$ROOT/outputs/iclr27_phase31/logs/${TAG}_f${fold}.stdout.log"; (cd "$ROOT"; CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. exec "$PY" scripts/iclr27_phase31/train_reranker.py --fold "$fold" --steps "$STEPS" --checkpoint-every 500 --tag "$TAG" --device cuda:0 --expected-physical-gpu "$gpu") >"$log" 2>&1 & pids+=("$!"); done
status=0; for p in "${pids[@]}"; do wait "$p" || status=1; done; exit "$status"
