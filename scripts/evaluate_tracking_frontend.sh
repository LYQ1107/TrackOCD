#!/usr/bin/env bash
set -euo pipefail
ROOT="/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT"
echo "== TrackEval summary =="
cat "$ROOT/outputs/iclr27_closure/tracking_eval/simowt/summary.csv"
echo "== track quality =="
cat "$ROOT/outputs/iclr27_closure/tracking_eval/simowt/track_quality_metrics.csv"
echo "== coverage =="
cat "$ROOT/outputs/iclr27_closure/tables/track_coverage_table.csv"
echo "== end-to-end =="
cat "$ROOT/outputs/iclr27_closure/end_to_end/coverage_aware_results.csv"
echo "== status =="
cat "$ROOT/runs/iclr27_closure/status.txt"
