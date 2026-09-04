#!/usr/bin/env bash
set -euo pipefail
TAG="${1:-b84sq_formal_v3}"
EPOCHS="${2:-15}"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
for fold in 0 1 2 3; do
  done_file="outputs/iclr27_phase84/completion/b84sq_${TAG}_f${fold}.done"
  launch_file="outputs/iclr27_phase84/completion/b84sq_${TAG}_f${fold}.launched"
  if [[ -f "$done_file" ]]; then
    continue
  fi
  if [[ -f "$launch_file" ]]; then
    echo "refusing relaunch of launched unfinished fold ${fold}" >&2
    exit 2
  fi
  "$PY" scripts/iclr27_phase84/train_b84s_balanced.py --fold "$fold" --tag "$TAG" --epochs "$EPOCHS"
done
printf 'B84S-Q supervisor complete tag=%s\n' "$TAG"
