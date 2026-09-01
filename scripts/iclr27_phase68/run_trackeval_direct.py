#!/usr/bin/env python3
"""Run the pinned TrackEval TAO metrics with an explicit string config.

The upstream CLI treats ``OUTPUT_FOLDER`` (whose default is ``None``) as an
``nargs`` list.  Direct construction avoids that parser bug and leaves the
vendored evaluator untouched.  NumPy aliases are process-local compatibility
shims only.
"""
from __future__ import annotations

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
    out = ROOT / "outputs/iclr27_phase68/metrics/ovtr_baseline/trackeval"
    out.mkdir(parents=True, exist_ok=True)
    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "PRINT_RESULTS": True,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": True,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
    })
    ds_cfg = trackeval.datasets.TAO.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(ROOT / "outputs/iclr27_phase68/trackeval/gt"),
        "TRACKERS_FOLDER": str(ROOT / "outputs/iclr27_phase68/trackeval/trackers"),
        "OUTPUT_FOLDER": str(out),
        "TRACKERS_TO_EVAL": ["OVTR_Q0"],
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "MAX_DETECTIONS": 300,
        "PRINT_CONFIG": True,
    })
    metrics_cfg = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset_list = [trackeval.datasets.TAO(ds_cfg)]
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    result = evaluator.evaluate(dataset_list, metrics_list)
    # Evaluator prints the authoritative table and writes TrackEval summaries.
    # Keep a tiny machine-readable completion note with the exact config.
    note = {
        "protocol": "trackocd_phase68_trackeval_tao",
        "gt_folder": ds_cfg["GT_FOLDER"],
        "trackers_folder": ds_cfg["TRACKERS_FOLDER"],
        "output_folder": ds_cfg["OUTPUT_FOLDER"],
        "tracker": "OVTR_Q0",
        "metrics": metrics_cfg["METRICS"],
        "numpy_compat": {"np.int": "int", "np.float": "float"},
        "result_type": type(result).__name__,
    }
    with (out / "run_note.json.tmp").open("w") as f:
        json.dump(note, f, indent=2)
    (out / "run_note.json.tmp").replace(out / "run_note.json")


if __name__ == "__main__":
    main()
