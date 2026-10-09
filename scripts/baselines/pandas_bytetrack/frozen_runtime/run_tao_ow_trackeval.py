#!/usr/bin/env python3
"""Run the pinned canonical TrackEval TAO-OW evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trackeval-root", type=Path, required=True)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--trackers-folder", type=Path, required=True)
    parser.add_argument("--tracker", default="bytetrack")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # The pinned canonical TAO_OW adapter predates NumPy 2 aliases. This
    # compatibility alias leaves the evaluator source unchanged and is
    # limited to this process.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    sys.path.insert(0, str(args.trackeval_root))
    import trackeval  # noqa: E402

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(args.gt_folder)
    dataset_config["TRACKERS_FOLDER"] = str(args.trackers_folder)
    dataset_config["TRACKERS_TO_EVAL"] = [args.tracker]
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["SPLIT_TO_EVAL"] = "val"
    dataset_config["SUBSET"] = "all"
    dataset_config["MAX_DETECTIONS"] = 300
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.TAO_OW(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    output_res, output_msg = evaluator.evaluate([dataset], metrics)

    combined = output_res["TAO_OW"][args.tracker]["COMBINED_SEQ"]["cls_comb_cls_av"]
    hota = combined["HOTA"]
    clear = combined["CLEAR"]
    identity = combined["Identity"]
    summary = {
        "status": "PASS",
        "dataset": "TAO_OW",
        "split": "val",
        "subset": "all",
        "class_mode": "class-agnostic",
        "tracker": args.tracker,
        "canonical_trackeval_root": str(args.trackeval_root),
        "metrics": {
            "HOTA_0": float(hota["HOTA(0)"]),
            "LocA_0": float(hota["LocA(0)"]),
            "AssocA_0": float(hota["AssA"][0]),
            "DetA_0": float(hota["DetA"][0]),
            "HOTA_mean": float(np.mean(hota["HOTA"])),
            "LocA_mean": float(np.mean(hota["LocA"])),
            "AssocA_mean": float(np.mean(hota["AssA"])),
            "DetA_mean": float(np.mean(hota["DetA"])),
            "IDF1": float(identity["IDF1"]),
            "MOTA": float(clear["MOTA"]),
            "IDSW": int(clear["IDSW"]),
            "Frag": int(clear["Frag"]),
        },
        "raw_combined": jsonable(combined),
        "evaluator_message": jsonable(output_msg),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
