#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-/home/lwr/anaconda3/envs/ovtr/bin/python}"
TAG="${1:-g1_formal}"
STEPS="${2:-2000}"
mkdir -p "${ROOT}/outputs/iclr27_phase76g/logs"
declare -a PIDS=()
declare -a FOLDS=()
declare -a GPUS=(4 5 6 7)

cleanup() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    for pid in "${PIDS[@]:-}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then kill -TERM "${pid}" 2>/dev/null || true; fi
    done
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

for fold in 0 1 2 3; do
  run="${TAG}_f${fold}"
  done_marker="${ROOT}/outputs/iclr27_phase76g/completion/${run}.done"
  launched_marker="${ROOT}/outputs/iclr27_phase76g/completion/${run}.launched"
  if [[ -f "${done_marker}" ]]; then echo "skip completed ${run}"; continue; fi
  if [[ -f "${launched_marker}" ]]; then echo "skip launched-without-done ${run}; explicit resume required"; continue; fi
  gpu="${GPUS[$fold]}"; log="${ROOT}/outputs/iclr27_phase76g/logs/${run}.log"
  echo "launch fold=${fold} gpu=${gpu} steps=${STEPS} log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${ROOT}" "${PY}" "${ROOT}/scripts/iclr27_phase76g/train_group_router_fold.py" --fold "${fold}" --steps "${STEPS}" --tag "${TAG}" --device cuda:0 --expected-physical-gpu "${gpu}" >"${log}" 2>&1 &
  PIDS+=("$!"); FOLDS+=("${fold}")
done

status=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"; fold="${FOLDS[$i]}"
  if wait "${pid}"; then echo "fold=${fold} pid=${pid} completed"; else rc=$?; echo "fold=${fold} pid=${pid} failed rc=${rc}" >&2; status=$rc; fi
done
exit "${status}"
