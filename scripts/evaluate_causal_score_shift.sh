#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
PY="/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python"
export PYTHONPATH="$ROOT"
echo "== audit status =="
cat "$ROOT/runs/router_audit/audit_gate.json" | "$PY" -c "import json,sys; g=json.load(sys.stdin); print(g['status'], g['selected_router'], g['corrected_outer_hmeans'])"
echo "== causal gate =="
"$PY" -c "import json; print(json.dumps(json.load(open('$ROOT/runs/causal_score_shift/causal_gate.json')), indent=1))"
echo "== causal summary (Pure Full) =="
"$PY" - <<'PY'
import csv
with open("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT/outputs/causal_score_shift/metrics/final_summary.csv") as f:
    for r in csv.DictReader(f):
        if r["protocol"] == "pure" and r["subset"] == "full":
            print(f"{r['method']} known={float(r['overall_known_acc_mean']):.4f} "
                  f"route={float(r['route_aware_novel_acc_mean']):.4f} "
                  f"recall={float(r['novel_routing_recall_mean']):.4f} "
                  f"cond={float(r['conditional_novel_acc_mean']):.4f} "
                  f"nmi={float(r['novel_only_nmi_mean']):.4f} "
                  f"ari={float(r['novel_only_ari_mean']):.4f} "
                  f"err={float(r['novel_count_abs_error_mean']):.1f}")
PY
echo "== status =="
cat "$ROOT/runs/causal_score_shift/status.txt"
