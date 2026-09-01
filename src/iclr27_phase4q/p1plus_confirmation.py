#!/usr/bin/env python3
"""P1+ confirmation control for Phase 4Q.

Trains a logistic calibration on *only*:
    current score, track age, hit count, disappear_time
using the frozen dev proposals, then reports the same dev/heldout
Novel-Recall-FP protocol. No query embedding / CIP / appearance /
semantic feature is allowed.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys_path = ROOT / "src" / "iclr27_phase4p"
import sys
sys.path.insert(0, str(sys_path))
from ovtr_main_eval import curve_metrics  # noqa: E402


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "video_id": int(r["video_id"]),
                "frame_id": int(r["frame_id"]),
                "proposal_local_id": int(r["proposal_local_id"]),
                "track_id": int(r["track_id"]),
                "score": float(r["score"]),
                "gt_role": r["gt_role"],
                "prior_hits": int(r["prior_hits"]),
            })
    return rows


def add_age_disappear(rows):
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(r["video_id"], r["track_id"])].append(i)
    first_frame = {}
    for key, idxs in by_track.items():
        idxs.sort(key=lambda i: (rows[i]["frame_id"], rows[i]["proposal_local_id"]))
        first_frame[key] = rows[idxs[0]]["frame_id"]
        prev = None
        for i in idxs:
            f = rows[i]["frame_id"]
            rows[i]["age"] = max(0, f - first_frame[key])
            rows[i]["disappear_time"] = (
                0 if prev is None else max(0, f - prev - 1))
            prev = f
    return rows


def feature_matrix(rows):
    X = np.array([[r["score"], r["age"], r["prior_hits"],
                   r["disappear_time"]] for r in rows], dtype=np.float64)
    return X


def target(rows):
    return np.array([1 if r["gt_role"] in ("known", "novel") else 0
                     for r in rows], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--heldout-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dev = add_age_disappear(load_rows(Path(args.dev_csv)))
    ho = add_age_disappear(load_rows(Path(args.heldout_csv)))
    X_dev, y_dev = feature_matrix(dev), target(dev)
    X_ho = feature_matrix(ho)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(dev), dtype=np.float64)
    for tr, va in skf.split(X_dev, y_dev):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_dev[tr], y_dev[tr])
        oof[va] = clf.predict_proba(X_dev[va])[:, 1]

    full = LogisticRegression(max_iter=2000)
    full.fit(X_dev, y_dev)
    ho_prob = full.predict_proba(X_ho)[:, 1]

    dev_adj = [dict(r) for r in dev]
    ho_adj = [dict(r) for r in ho]
    for r, p in zip(dev_adj, oof):
        r["score"] = float(p)
    for r, p in zip(ho_adj, ho_prob):
        r["score"] = float(p)

    n_frames_dev = len({(r["video_id"], r["frame_id"]) for r in dev})
    n_frames_ho = len({(r["video_id"], r["frame_id"]) for r in ho})
    m_dev = curve_metrics(dev_adj, n_frames_dev)
    m_ho = curve_metrics(ho_adj, n_frames_ho)
    report = {
        "mode": "p1plus",
        "features": ["score", "age", "hit_count", "disappear_time"],
        "train": "dev (5-fold OOF for dev; full-fit for heldout)",
        "dev": {**m_dev, "n_frames": n_frames_dev},
        "heldout": {**m_ho, "n_frames": n_frames_ho},
    }
    (out / "p1plus_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print("P1PLUS_DONE")


if __name__ == "__main__":
    main()
