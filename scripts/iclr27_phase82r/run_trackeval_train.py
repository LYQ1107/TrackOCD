#!/usr/bin/env python3
"""Run the vendored TrackEval TAO metrics on a Phase82R TRAIN export."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
import trackeval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    base = ROOT / "outputs/iclr27_phase82r/trackeval" / args.tag
    out = ROOT / "outputs/iclr27_phase82r/metrics/trackeval" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    ec = trackeval.Evaluator.get_default_eval_config()
    ec.update({"USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1, "PRINT_RESULTS": True,
               "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": True, "TIME_PROGRESS": False,
               "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True,
               "PLOT_CURVES": False, "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False})
    dc = trackeval.datasets.TAO.get_default_dataset_config()
    dc.update({"GT_FOLDER": str(base / "gt"), "TRACKERS_FOLDER": str(base / "trackers"),
               "OUTPUT_FOLDER": str(out), "TRACKERS_TO_EVAL": [args.tag],
               "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "MAX_DETECTIONS": 300,
               "PRINT_CONFIG": True, "SPLIT_TO_EVAL": "training"})
    evaluator = trackeval.Evaluator(ec)
    result = evaluator.evaluate([trackeval.datasets.TAO(dc)],
                                [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    note = {"protocol": "trackocd_phase82r_train_classagnostic_trackeval", "tag": args.tag,
            "gt_folder": dc["GT_FOLDER"], "trackers_folder": dc["TRACKERS_FOLDER"],
            "output_folder": dc["OUTPUT_FOLDER"], "metrics": ["HOTA", "CLEAR", "Identity"],
            "class_agnostic": True, "result_type": type(result).__name__}
    tmp = out / "run_note.json.tmp"
    tmp.write_text(json.dumps(note, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out / "run_note.json")
    print(json.dumps(note, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
