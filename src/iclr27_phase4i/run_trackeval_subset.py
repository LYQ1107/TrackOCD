"""TrackEval (HOTA/CLEAR/Identity) for Phase 4I subset trackers."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TRACKEVAL_ROOT = ROOT / "third_party" / "TrackEval"
sys.path.insert(0, str(TRACKEVAL_ROOT))

import trackeval  # noqa: E402

GT_FOLDER = Path(os.environ.get(
    "PHASE4L_TRACKEVAL_GT",
    ROOT / "outputs" / "iclr27_phase3a" / "trackeval" / "gt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trackers-folder", required=True, type=Path)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["USE_PARALLEL"] = False
    eval_config["PRINT_ONLY_COMBINED"] = True
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PLOT_CURVES"] = False
    dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
    dataset_config["GT_FOLDER"] = str(GT_FOLDER)
    dataset_config["TRACKERS_FOLDER"] = str(args.trackers_folder)
    dataset_config["TRACKERS_TO_EVAL"] = args.names
    dataset_config["TRACKER_SUB_FOLDER"] = "data"
    dataset_config["SUBSET"] = "all"
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.TAO_OW(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(),
               trackeval.metrics.Identity()]
    output_res, _ = evaluator.evaluate([dataset], metrics)

    def conv(v):
        if isinstance(v, np.ndarray):
            return [conv(x) for x in v.tolist()]
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        return v

    simplified = {}
    for tracker in args.names:
        combined = output_res["TAO_OW"][tracker]["COMBINED_SEQ"]["cls_comb_cls_av"]
        simplified[tracker] = {
            metric_name: {k: conv(v) for k, v in combined[metric_name].items()}
            for metric_name in combined
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(simplified, indent=1))
    flat = {}
    for tracker in args.names:
        c = simplified[tracker]
        flat[tracker] = {
            "HOTA": c["HOTA"]["HOTA"],
            "DetA": c["HOTA"]["DetA"],
            "AssA": c["HOTA"]["AssA"],
            "LocA": c["HOTA"]["LocA(0)"],
            "IDF1": c["Identity"]["IDF1"],
            "MOTA": c["CLEAR"]["MOTA"],
            "MOTP": c["CLEAR"]["MOTP"],
            "IDSW": c["CLEAR"]["IDSW"],
            "Frag": c["CLEAR"]["Frag"],
            "MT": c["CLEAR"]["MT"],
            "PT": c["CLEAR"]["PT"],
            "ML": c["CLEAR"]["ML"],
        }
    print(json.dumps(flat, indent=1))


if __name__ == "__main__":
    main()
