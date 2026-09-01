#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
export PYTHONPATH="$ROOT"
echo "== tests =="
"$PY" "$ROOT/tests/dinov3_bakeoff/test_dinov3_bakeoff.py" > /dev/null && echo "18/18 passed"
echo "== status =="
cat "$ROOT/runs/dinov3_bakeoff/status.txt"
echo "== backbone gate =="
"$PY" -c "import json; print(json.dumps(json.load(open('$ROOT/runs/dinov3_bakeoff/backbone_gate.json')), indent=1))"
echo "== summary (Pure Full) =="
"$PY" - <<'PY'
import csv
with open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/dinov3_bakeoff/metrics/backbone_summary.csv") as f:
    for r in csv.DictReader(f):
        if r["protocol"] == "pure" and r["subset"] == "full" and r["method"] in ("V0", "V2", "O0", "O1"):
            print(f"{r['method']} all={float(r['all_track_acc_mean']):.4f} known={float(r['overall_known_acc_mean']):.4f} "
                  f"route={float(r['route_aware_novel_acc_mean']):.4f} cond={float(r['conditional_novel_acc_mean']):.4f} "
                  f"nmi={float(r['novel_only_nmi_mean']):.4f} ari={float(r['novel_only_ari_mean']):.4f} "
                  f"err={float(r['novel_count_abs_error_mean']):.1f}")
PY
