#!/usr/bin/env python3
"""Run TrackEval HOTA/CLEAR/Identity on the Phase 3A 20-video subset."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKEVAL_ROOT = PROJECT_ROOT / "third_party" / "TrackEval"
sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402

GT_FOLDER = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "trackeval" / "gt"
TRACKERS_FOLDER = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "trackeval" / "trackers"
OUT_JSON = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "trackeval" / "results.json"
TRACKERS = ["original", "uninstrumented", "instrumented", "offline_replay"]


def main() -> None:
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["DISPLAY_LESS_PROGRESS"] = True

    dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(GT_FOLDER)
    dataset_config["TRACKERS_FOLDER"] = str(TRACKERS_FOLDER)
    dataset_config["TRACKERS_TO_EVAL"] = TRACKERS
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["SUBSET"] = "all"

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.TAO_OW(dataset_config)
    metrics = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]
    output_res, _ = evaluator.evaluate([dataset], metrics)

    def conv(v):
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        return v

    simplified = {}
    for tracker in TRACKERS:
        combined = output_res["TAO_OW"][tracker]["COMBINED_SEQ"]["cls_comb_cls_av"]
        simplified[tracker] = {
            metric_name: {k: conv(v) for k, v in combined[metric_name].items()}
            for metric_name in combined
        }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(simplified, indent=1))
    print(json.dumps(simplified, indent=1))


if __name__ == "__main__":
    main()
