#!/usr/bin/env python3
"""Compute CLEAR and Identity metrics for TAO_OW using the official TrackEval
metric classes over the same preprocessed sequence data (run with the SimOWT
env, numpy 1.x). Also recomputes HOTA to verify Phase 1 reproduction."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT / "third_party/TrackEval"))

import trackeval  # noqa: E402
from trackeval.datasets import TAO_OW  # noqa: E402
from trackeval.metrics import HOTA, CLEAR, Identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="all")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_path = ROOT / "third_party/TrackEval"
    dataset_config = {
        "GT_FOLDER": str(code_path / "data/gt/tao/tao_validation"),
        "TRACKERS_FOLDER": str(code_path / "data/trackers/tao/tao_validation"),
        "OUTPUT_FOLDER": str(out_dir),
        "TRACKERS_TO_EVAL": ["simowt"],
        "SPLIT_TO_EVAL": "val",
        "SUBSET": args.subset,
        "MAX_DETECTIONS": 0,
        "PRINT_CONFIG": False,
    }
    ds = TAO_OW(dataset_config)
    tracker = ds.tracker_list[0]
    metrics = [HOTA(), CLEAR(), Identity()]
    all_res = {m.get_name(): {} for m in metrics}
    for seq in ds.seq_list:
        raw = ds.get_raw_seq_data(tracker, seq)
        for cls in ds.class_list:
            data = ds.get_preprocessed_seq_data(raw, cls)
            for m in metrics:
                all_res[m.get_name()][seq] = m.eval_sequence(data)
    combined = {}
    for m in metrics:
        combined[m.get_name()] = m.combine_sequences(all_res[m.get_name()])
    # write outputs
    import numpy as np

    def sanitize(v):
        if isinstance(v, np.ndarray):
            return [sanitize(x) for x in v.tolist()]
        if hasattr(v, "item"):
            try:
                return float(v)
            except TypeError:
                return v.tolist()
        return v

    rows = {}
    for name in combined:
        rows[name] = {k: sanitize(v) for k, v in combined[name].items()}
    (out_dir / "combined.json").write_text(json.dumps(rows, indent=2))
    # per-metric CSVs
    for name, vals in rows.items():
        with open(out_dir / f"{name.lower()}_metrics.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in vals.items():
                w.writerow([k, v])
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
