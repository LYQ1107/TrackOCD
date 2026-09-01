#!/usr/bin/env bash
set -euo pipefail

# Bounded Phase71 supervisor.  It never starts a unit with an existing
# .launched marker, maps fold→GPU4..7, and writes completion atomically.
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
MODE="${1:?usage: run_tco_supervisor.sh smoke|targeted|formal}"
TAG="${2:?usage: run_tco_supervisor.sh MODE TAG}"
COMP="$ROOT/outputs/iclr27_phase71/completion"
RUN="$ROOT/outputs/iclr27_phase71/runs/$TAG"
mkdir -p "$COMP" "$RUN"

declare -A GPUS=([0]=4 [1]=5 [2]=6 [3]=7)
declare -A PIDS=()

if [[ "$MODE" == "formal" ]]; then
  FOLDS=(0 1 2 3)
else
  # Smoke and targeted are deliberately single-fold (GPU4) checks.
  FOLDS=(0)
fi

for fold in "${FOLDS[@]}"; do
  done_marker="$COMP/${TAG}_f${fold}.done"
  launch_marker="$COMP/${TAG}_f${fold}.launched"
  if [[ -e "$done_marker" ]]; then
    echo "PHASE71_SKIP_DONE fold=$fold tag=$TAG"
    continue
  fi
  if [[ -e "$launch_marker" ]]; then
    echo "PHASE71_REFUSE_RELAUNCH fold=$fold tag=$TAG (launched marker exists)" >&2
    exit 3
  fi
  tmp="${launch_marker}.tmp.$$"
  printf 'pid_pending mode=%s fold=%s gpu=%s started_utc=%s\n' "$MODE" "$fold" "${GPUS[$fold]}" "$(date -u +%FT%TZ)" > "$tmp"
  mv "$tmp" "$launch_marker"
  log="$RUN/fold_${fold}.log"
  ( set +e; \
    export PHASE71_FOLD="$fold" PHASE71_TAG="$TAG" PHASE71_MODE="$MODE" CUDA_VISIBLE_DEVICES="${GPUS[$fold]}"; \
    "$ROOT/scripts/iclr27_phase71/run_tco_unit.sh" >"$log" 2>&1; \
    rc=$?; \
    if [[ "$rc" -eq 0 && -s "$RUN/fold_${fold}/checkpoint.pth" ]]; then \
      printf 'complete mode=%s fold=%s gpu=%s finished_utc=%s\n' "$MODE" "$fold" "${GPUS[$fold]}" "$(date -u +%FT%TZ)" > "${done_marker}.tmp"; \
      mv "${done_marker}.tmp" "$done_marker"; \
    else \
      printf 'failed mode=%s fold=%s gpu=%s rc=%s finished_utc=%s\n' "$MODE" "$fold" "${GPUS[$fold]}" "$rc" "$(date -u +%FT%TZ)" > "${COMP}/${TAG}_f${fold}.failed"; \
      exit "$rc"; \
    fi \
  ) &
  PIDS[$fold]=$!
done

rc=0
for fold in "${!PIDS[@]}"; do
  pid="${PIDS[$fold]}"
  if ! wait "$pid"; then rc=1; fi
done
exit "$rc"
