#!/usr/bin/env bash
set -euo pipefail

# One bounded supervisor for the pre-registered C0_CONTINUE/C1_SUPPORT
# comparison. Each group uses the same four frozen fold checkpoints and
# restored optimizer/sampler/RNG state; only support_mode differs.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="/home/lwr/anaconda3/envs/ovtr/bin/python"
OUT="$ROOT/outputs/iclr27_phase88"
SHARED="/data2/usr_for_deadline/trackocd_phase88/shared_features"
LOG="$OUT/logs"
mkdir -p "$LOG"
declare -a GPUS=(0 1 2 3)
declare -a PIDS=()

preflight="$OUT/audit/c0_c1_fair_preflight.txt"
{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "free_h:"; free -h
  echo "process_count=$(ps -e --no-headers | wc -l)"
  echo "nvidia_smi:"; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
} >"$preflight"

launch_group() {
  local mode="$1"
  local support_flag=""
  local group_status="$OUT/audit/c0_c1_${mode}_launch.json"
  if [[ "$mode" == "c1_support_fix1" ]]; then support_flag="--support-mode"; fi
  PIDS=()
  for fold in 0 1 2 3; do
    local tag="${mode}_f${fold}"
    local done="$OUT/completion/${tag}.done"
    local launched="$OUT/completion/${tag}.launched"
    if [[ -f "$done" ]]; then
      echo "skip_done $tag"
      continue
    fi
    if [[ -f "$launched" ]]; then
      echo "refusing_relaunch_launched_without_done $tag" >&2
      return 2
    fi
    local init="$OUT/checkpoints/c0v2_fix2_formal_f${fold}.pt"
    [[ -f "$init" ]] || { echo "missing frozen init $init" >&2; return 2; }
    local gpu="${GPUS[$fold]}"
    "$PY" "$ROOT/scripts/iclr27_phase88/train_controller.py" \
      --fold "$fold" --device "cuda:${gpu}" --updates 30000 \
      --tag "$tag" --seed 88002 --checkpoint-interval 2000 \
      --event-tag fix2 --memmap-root "$SHARED" \
      --resume-checkpoint "$init" $support_flag \
      >"$LOG/${tag}.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "launched tag=$tag fold=$fold gpu=$gpu pid=$pid"
  done
  {
    echo "group=$mode"
    echo "post_launch_timestamp=$(date --iso-8601=seconds)"
    echo "post_launch_free_h:"; free -h
    echo "post_launch_nvidia_smi:"; nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
  } >>"$group_status"
  local status=0
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then status=1; fi
  done
  if [[ "$status" != 0 ]]; then
    echo "training_group_failed mode=$mode status=$status" >&2
    return "$status"
  fi
  for fold in 0 1 2 3; do
    local tag="${mode}_f${fold}"
    local done="$OUT/completion/${tag}.done"
    [[ -f "$done" ]] || { echo "missing training done $tag" >&2; return 2; }
    local ckpt="$OUT/checkpoints/${tag}.pt"
    local val_tag="${tag}_val"
    "$PY" "$ROOT/scripts/iclr27_phase88/evaluate_checkpoint_sharded.py" \
      --fold "$fold" --checkpoint "$ckpt" --tag "$val_tag" \
      --event-tag fix2 --shard-size 250 --device cuda:0 \
      --memmap-root "$SHARED" $support_flag \
      >"$LOG/${val_tag}.log" 2>&1
    [[ -f "$OUT/validation/$val_tag/final.done" ]] || { echo "missing validation done $val_tag" >&2; return 2; }
  done
  echo "group_complete mode=$mode"
}

launch_group c0_continue_fix1
launch_group c1_support_fix1

status_file="$OUT/audit/c0_c1_fair_supervisor_status.json"
OUT="$OUT" "$PY" - "$status_file" <<'PY'
import datetime as dt, json, os, sys
from pathlib import Path
out = Path(os.environ["OUT"])
status_file = Path(sys.argv[1])
rows = []
for mode in ("c0_continue_fix1", "c1_support_fix1"):
    for fold in range(4):
        tag = f"{mode}_f{fold}"
        train_metrics = out / "metrics" / f"{tag}.json"
        val_metrics = out / "validation" / f"{tag}_val" / "final_metrics.json"
        rows.append({
            "mode": mode, "fold": fold, "tag": tag,
            "train_metrics": str(train_metrics.resolve()),
            "validation_metrics": str(val_metrics.resolve()),
            "done": (out / "completion" / f"{tag}.done").exists(),
            "validation_done": (out / "validation" / f"{tag}_val" / "final.done").exists(),
        })
payload = {
    "schema_version": "trackocd.phase88.c0_c1_fair_supervisor.v1",
    "status": "COMPLETE", "rows": rows,
    "public_dev_q1_sealed_accessed": False,
    "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
}
tmp = status_file.with_name(f".{status_file.name}.tmp.{os.getpid()}")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, status_file)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
