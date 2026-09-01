#!/usr/bin/env bash
# TrackOCD-v1.0 evaluation entry point.
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
export PYTHONPATH="$ROOT"
echo "== unit tests =="
"$PY" "$ROOT/tests/test_trackocd_evaluator.py" > /dev/null && echo "9/9 passed"
echo "== corrected baseline summary =="
"$PY" - <<'PY'
import csv
with open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/trackocd_v1/metrics/corrected_baseline_summary.csv") as f:
    for r in csv.DictReader(f):
        if r["subset"] == "full":
            print(f"{r['protocol']:11s} {r['method']} all={float(r['all_track_acc_mean']):.4f} "
                  f"known={float(r['overall_known_acc_mean']):.4f} "
                  f"route_novel={float(r['route_aware_novel_acc_mean']):.4f} "
                  f"cond_novel={float(r['conditional_novel_acc_mean']):.4f}")
PY
echo "== architecture decision =="
head -12 "$ROOT/docs/trackocd_v1/ARCHITECTURE_DECISION.md"
echo "== status =="
cat "$ROOT/runs/trackocd_v1/status.txt"
