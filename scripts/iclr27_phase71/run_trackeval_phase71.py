#!/usr/bin/env python3
"""Run vendored TrackEval for all Phase71 serial-validation fold trackers."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
import trackeval  # noqa: E402

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="formal1_tco_serial"); args = ap.parse_args()
    root = ROOT / "outputs/iclr27_phase71/validation" / args.tag
    out = ROOT / "outputs/iclr27_phase71/metrics" / args.tag / "trackeval"
    out.mkdir(parents=True, exist_ok=True)
    ec = trackeval.Evaluator.get_default_eval_config(); ec.update({"USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1, "PRINT_RESULTS": True, "PRINT_ONLY_COMBINED": True, "PRINT_CONFIG": True, "TIME_PROGRESS": False, "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": True, "OUTPUT_DETAILED": True, "PLOT_CURVES": False, "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False})
    dc = trackeval.datasets.TAO.get_default_dataset_config(); dc.update({"GT_FOLDER": str(root / "trackeval/gt"), "TRACKERS_FOLDER": str(root / "trackeval/trackers"), "OUTPUT_FOLDER": str(out), "TRACKERS_TO_EVAL": [f"fold_{i}" for i in range(4)], "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "MAX_DETECTIONS": 300, "PRINT_CONFIG": True})
    metrics_cfg = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
    evaluator = trackeval.Evaluator(ec)
    result = evaluator.evaluate([trackeval.datasets.TAO(dc)], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])
    note = {"protocol": "trackocd_phase71_trackeval_tao", "tag": args.tag, "config": {"dataset": dc, "metrics": metrics_cfg}, "result_type": type(result).__name__}
    tmp = out / "run_note.json.tmp"; tmp.write_text(json.dumps(note, indent=2)); tmp.replace(out / "run_note.json")

if __name__ == "__main__": main()
