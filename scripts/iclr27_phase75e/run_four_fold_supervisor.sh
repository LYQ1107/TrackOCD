#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
TAG="${1:-phase75e_formal}"
STEPS="${2:-15000}"
OUT="${ROOT}/outputs/iclr27_phase75e"
mkdir -p "${OUT}/logs" "${OUT}/completion"

# Explicit four-device mapping.  The caller records a fresh nvidia-smi
# preflight; this script never kills or modifies an unrelated process.
GPUS=(4 5 6 7)
for g in "${GPUS[@]}"; do
  if ! nvidia-smi -i "${g}" --query-gpu=index,memory.used --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "GPU ${g} is not queryable" >&2
    exit 2
  fi
done

declare -a PIDS=()
for fold in 0 1 2 3; do
  run="${TAG}_f${fold}"
  if [[ -e "${OUT}/completion/${run}.done" ]]; then
    echo "skip completed ${run}"
    continue
  fi
  if [[ -e "${OUT}/completion/${run}.launched" ]]; then
    echo "refusing duplicate launched unit ${run}" >&2
    exit 3
  fi
  gpu="${GPUS[$fold]}"
  log="${OUT}/logs/${run}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}" "${PYTHON}" "${ROOT}/scripts/iclr27_phase75e/train_adapter_fold.py" \
    --fold "${fold}" --steps "${STEPS}" --tag "${TAG}" --device cuda:0 --expected-physical-gpu "${gpu}" >"${log}" 2>&1 &
  PIDS+=("$!")
done

# One blocking wait for all launched workers; there is deliberately no poll loop.
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
