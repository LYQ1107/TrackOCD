#!/usr/bin/env bash
set -euo pipefail

PROJ=/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT
cd "$PROJ"
DISCOVERY_PY=/home/lwr/anaconda3/envs/ocd_ovmot_discovery/bin/python
export PYTHONPATH="$PROJ"

if [[ -f runs/arch1_main/09_trackeval.done ]]; then
  echo "[skip] TrackEval already done"
else
  "$DISCOVERY_PY" "$PROJ/scripts/merge_simowt_output.py" \
    --input-dir "$PROJ/runs" \
    --output-json outputs/simowt/val_predictions.json \
    --stream-jsonl data/tao_ow_ocd_v1/public/pred_track_stream.jsonl
  mkdir -p third_party/TrackEval/data/trackers/tao/tao_training/simowt/data
  cp outputs/simowt/val_predictions.json third_party/TrackEval/data/trackers/tao/tao_training/simowt/data/pred.json
  for subset in known unknown; do
    (cd third_party/TrackEval && LD_LIBRARY_PATH=/home/lwr/anaconda3/lib:/usr/local/cuda-11.6/lib64 \
      /home/lwr/anaconda3/envs/ocd_ovmot_simowt/bin/python scripts/run_tao_ow.py \
      --USE_PARALLEL False --METRICS HOTA --TRACKERS_TO_EVAL simowt --SUBSET "$subset" \
      --GT_FOLDER data/gt/tao/tao_training --TRACKERS_FOLDER data/trackers/tao/tao_training)
  done
fi

"$DISCOVERY_PY" src/evaluation/track_matching.py
"$DISCOVERY_PY" src/evaluation/summarize.py
echo "Summary written to outputs/metrics/summary.csv"
