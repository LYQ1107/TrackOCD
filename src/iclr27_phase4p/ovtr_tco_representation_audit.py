#!/usr/bin/env python3
"""Audit TCO logit/score distributions against GT roles from a labeled
OVTR proposal CSV (produced by ovtr_main_eval.py)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_VAL = ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) +
          (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-json", required=True)
    ap.add_argument("--proposals-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    val = json.loads(TAO_VAL.read_text())
    img_id_by_name = {im["file_name"]: im["id"] for im in val["images"]}

    prop_rows = list(csv.DictReader(open(args.proposals_csv)))
    prop_by_key = {}
    for r in prop_rows:
        key = (int(r["image_id"]), int(r["track_id"]))
        prop_by_key.setdefault(key, r)

    stats_path = Path(args.stats_json)
    first = stats_path.read_text()[:1]
    if first != "[":
        raise ValueError("expected JSON array stats")

    with open(stats_path) as f:
        sample = f.read(200)
    has_bbox = '"bbox"' in sample

    groups = defaultdict(list)
    unmatched = 0
    hit_groups = defaultdict(list)
    if has_bbox:
        import ijson
        proposals_by_img = defaultdict(list)
        for r in prop_rows:
            proposals_by_img[int(r["image_id"])].append(r)
        with open(stats_path, "rb") as f:
            for s in ijson.items(f, "item"):
                bb = [float(v) for v in s["bbox"]]
                if (bb[2] - bb[0]) <= 0 or (bb[3] - bb[1]) <= 0:
                    continue
                img_id = img_id_by_name.get(s["file_path"])
                if img_id is None:
                    unmatched += 1
                    continue
                best = None
                best_iou = 0.5
                for prop in proposals_by_img.get(img_id, []):
                    pb = json.loads(prop["bbox_xyxy"])
                    v = iou(bb, pb)
                    if v >= best_iou:
                        best_iou, best = v, prop
                if best is None:
                    unmatched += 1
                    continue
                hit = int(s["hit_count"])
                if hit <= 1:
                    age = "new"
                elif hit == 2:
                    age = "persistent1"
                elif hit == 3:
                    age = "persistent2"
                else:
                    age = "persistent3plus"
                key = (age, best["gt_role"])
                groups[key].append((float(s["score"]), float(s["tco_logit"]),
                                    int(s["disappear_time"]), hit))
    else:
        # Legacy stats without bbox: only hit-count aggregate is possible.
        import ijson
        with open(stats_path, "rb") as f:
            for s in ijson.items(f, "item"):
                hit = int(s["hit_count"])
                if hit <= 1:
                    age = "new"
                elif hit == 2:
                    age = "persistent1"
                else:
                    age = "persistent2plus"
                hit_groups[age].append(
                    (float(s["score"]), float(s["tco_logit"]),
                     float(s["disappear_time"])))

    out = {"unmatched_stats": unmatched, "groups": {}}
    for (age, role), vals in sorted(groups.items()):
        arr = np.asarray(vals, dtype=np.float64)
        out["groups"][f"{age}/{role}"] = {
            "n": len(vals),
            "score_mean": float(arr[:, 0].mean()),
            "score_p10": float(np.percentile(arr[:, 0], 10)),
            "score_p90": float(np.percentile(arr[:, 0], 90)),
            "tco_logit_mean": float(arr[:, 1].mean()),
            "tco_logit_p10": float(np.percentile(arr[:, 1], 10)),
            "tco_logit_p90": float(np.percentile(arr[:, 1], 90)),
            "disappear_time_mean": float(arr[:, 2].mean()),
            "hit_count_mean": float(arr[:, 3].mean()),
        }
        print(f"{age}/{role}: n={len(vals):6d} "
              f"score={arr[:,0].mean():.4f} "
              f"tco={arr[:,1].mean():+.4f}")

    if hit_groups:
        out["hit_count_groups"] = {}
        for age, vals in sorted(hit_groups.items()):
            arr = np.asarray(vals, dtype=np.float64)
            out["hit_count_groups"][age] = {
                "n": len(vals),
                "score_mean": float(arr[:, 0].mean()),
                "tco_logit_mean": float(arr[:, 1].mean()),
            }
            print(f"hit-group {age}: n={len(vals)} "
                  f"score={arr[:,0].mean():.4f} "
                  f"tco={arr[:,1].mean():+.4f}")

    # Key separation: persistent valid vs persistent FP.
    def collect(age_prefix, roles):
        vals = []
        for (age, role), v in groups.items():
            if age.startswith(age_prefix) and role in roles:
                vals.extend(x[1] for x in v)
        return np.asarray(vals) if vals else np.zeros(0)

    pos = collect("persistent", ("known", "novel"))
    neg = collect("persistent", ("fp",))
    if len(pos) and len(neg):
        out["separation"] = {
            "persistent_valid_mean_logit": float(pos.mean()),
            "persistent_fp_mean_logit": float(neg.mean()),
            "mean_diff": float(pos.mean() - neg.mean()),
        }
        print("separation persistent valid - fp:",
              out["separation"])

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("TCO_REPRESENTATION_AUDIT_DONE")


if __name__ == "__main__":
    main()
