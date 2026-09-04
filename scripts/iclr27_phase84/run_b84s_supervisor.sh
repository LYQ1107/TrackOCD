#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
TAG="${1:-formal_r1}"
EPOCHS="${2:-15}"
for fold in 0 1 2 3; do
  done_marker="$ROOT/outputs/iclr27_phase84/completion/b84s_${TAG}_f${fold}.done"
  launch_marker="$ROOT/outputs/iclr27_phase84/completion/b84s_${TAG}_f${fold}.launched"
  if [[ -f "$done_marker" ]]; then continue; fi
  if [[ -f "$launch_marker" ]]; then echo "unfinished launched fold $fold; refusing duplicate" >&2; exit 17; fi
  PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/iclr27_phase84/train_source_conditioned.py" --fold "$fold" --tag "$TAG" --epochs "$EPOCHS"
done
echo "B84S supervisor complete tag=$TAG"
