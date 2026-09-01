#!/usr/bin/env python3
"""Run pinned TrackEval TAO metrics for one Phase69 fold.

This Phase69-only wrapper does not modify vendored TrackEval.  NumPy aliases
are process-local compatibility shims for the old parser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
import trackeval  # noqa: E402


def atomic_json(path: pathlib.Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--pred-json", type=pathlib.Path, required=True)
    ap.add_argument("--gt-json", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    if not args.pred_json.exists() or not args.gt_json.exists():
        raise FileNotFoundError(f"missing prediction/GT: {args.pred_json} {args.gt_json}")
    tracker_root = args.out_dir / "trackers"
    tracker_data = tracker_root / f"PHASE69_F{args.fold}" / "data"
    gt_root = args.out_dir / "gt"
    result_root = args.out_dir / "results"
    tracker_data.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    pred_link = tracker_data / "tao_track.json"
    gt_link = gt_root / "validation.json"
    for dst, src in ((pred_link, args.pred_json.resolve()), (gt_link, args.gt_json.resolve())):
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and os.path.realpath(dst) == str(src):
                continue
            raise RuntimeError(f"refusing to overwrite existing TrackEval link {dst}")
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(str(src), str(tmp))
        os.replace(tmp, dst)

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
        "GT_FOLDER": str(gt_root),
        "TRACKERS_FOLDER": str(tracker_root),
        "OUTPUT_FOLDER": str(result_root),
        "TRACKERS_TO_EVAL": [f"PHASE69_F{args.fold}"],
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "MAX_DETECTIONS": 300,
        "PRINT_CONFIG": True,
    })
    evaluator = trackeval.Evaluator(eval_cfg)
    result = evaluator.evaluate(
        [trackeval.datasets.TAO(ds_cfg)],
        [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()],
    )
    atomic_json(args.out_dir / "run_note.json", {
        "protocol": "trackocd_phase69_ovtr_trackeval_tao",
        "fold": args.fold,
        "prediction": str(args.pred_json.resolve()),
        "prediction_sha256": hashlib.sha256(args.pred_json.read_bytes()).hexdigest(),
        "gt": str(args.gt_json.resolve()),
        "tracker": f"PHASE69_F{args.fold}",
        "metrics": ["HOTA", "CLEAR", "Identity"],
        "result_type": type(result).__name__,
        "numpy_compat": {"np.int": "int", "np.float": "float"},
    })


if __name__ == "__main__":
    main()
