#!/usr/bin/env bash
# Architecture 1.5 evaluation entry point.
# Usage: bash scripts/evaluate_arch1_5.sh
set -euo pipefail

ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
export PYTHONPATH="$ROOT"

echo "== Stage A summary =="
"$PY" "$ROOT/src/ocd_v2/summarize_stage_a.py"

echo "== Stage A gate result =="
cat "$ROOT/runs/arch1_5/stage_a_gate.json"

echo "== Track audit summary =="
"$PY" - <<'PY'
import json
s = json.load(open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/arch1_5/track_audit/pred_track_stats.json"))
print(json.dumps(s["track_stats"], indent=1)[:1500])
PY

echo "== Final status =="
cat "$ROOT/runs/arch1_5/final_status.txt"
