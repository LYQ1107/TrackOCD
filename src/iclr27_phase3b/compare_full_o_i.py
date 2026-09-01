"""Full 988-video O vs I writer non-interference validation."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
O_DIR = ROOT / "runs"
I_DIR = ROOT / "outputs" / "iclr27_phase3b" / "full_export" / "instrumented_online"
OUT = ROOT / "outputs" / "iclr27_phase3b" / "fidelity"


def norm(recs):
    return sorted(recs, key=lambda a: (a.get("track_id", -1), tuple(a.get("bbox", [])), a.get("score", 0)))


def main():
    o_files = {
        (p.name.removeprefix("simowt_inference") if p.name.startswith("simowt_inference") else p.name): p
        for p in O_DIR.glob("simowt_inference*.json")
    }
    i_files = {p.name: p for p in I_DIR.glob("*.json")}
    print("O files", len(o_files), "I files", len(i_files))
    common = sorted(set(o_files) & set(i_files))
    print("common", len(common))
    rows = []
    total_preds = 0
    exact_preds = 0
    track_agree = 0
    count_mismatch_frames = 0
    geom_mismatch_dets = 0
    max_b = 0.0
    max_s = 0.0
    for name in common:
        o = norm(json.loads(o_files[name].read_text()))
        i = norm(json.loads(i_files[name].read_text()))
        total_preds += len(o)
        if len(o) != len(i):
            count_mismatch_frames += 1
            continue
        exact = 0
        for x, y in zip(o, i):
            same_box = x["bbox"] == y["bbox"]
            same_score = abs(x["score"] - y["score"]) <= 1e-6
            same_track = x["track_id"] == y["track_id"]
            exact += int(same_box and same_score and same_track)
            track_agree += int(same_track)
            if not same_box:
                geom_mismatch_dets += 1
                for p, q in zip(x["bbox"], y["bbox"]):
                    max_b = max(max_b, abs(p - q))
            max_s = max(max_s, abs(x["score"] - y["score"]))
        exact_preds += exact
    rows.append({
        "o_files": len(o_files), "i_files": len(i_files), "common_files": len(common),
        "total_predictions": total_preds, "exact_predictions": exact_preds,
        "geometry_exact_rate": exact_preds / total_preds if total_preds else 1.0,
        "canonical_track_agreement": track_agree / total_preds if total_preds else 1.0,
        "count_mismatch_frames": count_mismatch_frames,
        "geometry_mismatch_detections": geom_mismatch_dets,
        "max_bbox_error": max_b, "max_score_error": max_s,
    })
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "full_o_vs_i.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows[0], indent=1))


if __name__ == "__main__":
    main()
