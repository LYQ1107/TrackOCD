"""Phase 4U cross-track semantic data capacity + raw crop audit.

Reads the real Q1 tracker-induced TRAIN stream (proposals.csv + feats.npz)
and reports, per supported-known category: physical GT track counts, matched
predicted tracklets, videos, fragments, prefix lengths, and same-video vs
cross-video positive-pair capacity. Also verifies that every proposal row can
be reconstructed as a raw crop (frames + bbox + frame index).
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN = set(json.loads((ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json").read_text()))


def load_rows() -> list[dict]:
    rows = []
    with open(ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "proposals.csv") as f:
        for r in csv.DictReader(f):
            r = dict(r)
            for k in ("video_id", "frame_id", "image_id", "track_id", "gt_category_id", "gt_track_id", "prior_hits", "age", "gap"):
                r[k] = int(r[k])
            r["score"] = float(r["score"])
            r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
            r["gt_role"] = r["gt_role"]
            rows.append(r)
    return rows


def main():
    rows = load_rows()
    known_rows = [r for r in rows if r["gt_role"] == "known"]
    print("total rows", len(rows), "known rows", len(known_rows))

    # matched predicted tracklets per category
    tl = defaultdict(list)  # cat -> [(video_id, track_id)]
    for r in known_rows:
        tl[r["gt_category_id"]].append((r["video_id"], r["track_id"]))
    tl = {c: sorted(set(v)) for c, v in tl.items()}
    tl_len = defaultdict(list)  # cat -> tracklet lengths
    by_key = defaultdict(list)
    for r in known_rows:
        by_key[(r["video_id"], r["track_id"])].append(r)
    for key, rs in by_key.items():
        cat = Counter(int(r["gt_category_id"]) for r in rs).most_common(1)[0][0]
        tl_len[cat].append(len(rs))

    # physical GT tracks + fragments per GT track
    gt = defaultdict(set)  # cat -> set of (video_id, gt_track_id)
    frag = defaultdict(Counter)  # cat -> fragments per gt track
    for r in known_rows:
        c = r["gt_category_id"]
        gt[c].add((r["video_id"], r["gt_track_id"]))
        frag[c][(r["video_id"], r["gt_track_id"])] += 1

    # positive pair capacity, same-video vs cross-video
    cats = sorted(tl)
    capacity = {}
    for c in cats:
        keys = tl[c]
        by_vid = defaultdict(list)
        for vid, tid in keys:
            by_vid[vid].append((vid, tid))
        same, cross = 0, 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i][0] == keys[j][0]:
                    same += 1
                else:
                    cross += 1
        capacity[c] = {
            "n_pred_tracklets": len(keys),
            "n_videos": len(by_vid),
            "n_gt_physical_tracks": len(gt[c]),
            "fragments_per_gt": {
                "min": min(frag[c].values()), "median": float(np.median(list(frag[c].values()))),
                "max": max(frag[c].values())},
            "tracklet_len": {
                "min": min(tl_len[c]), "median": float(np.median(tl_len[c])),
                "p90": float(np.percentile(tl_len[c], 90)), "max": max(tl_len[c])},
            "same_video_pairs": same,
            "cross_video_pairs": cross,
        }

    # thresholds
    for th in (2, 3, 5, 10, 20):
        print(f"categories with >= {th} predicted tracklets:",
              sum(1 for c in cats if capacity[c]["n_pred_tracklets"] >= th))
    print("categories with >= 2 GT physical tracks:",
          sum(1 for c in cats if capacity[c]["n_gt_physical_tracks"] >= 2))
    print("categories with >= 2 videos:",
          sum(1 for c in cats if capacity[c]["n_videos"] >= 2))
    total_cross = sum(capacity[c]["cross_video_pairs"] for c in cats)
    total_same = sum(capacity[c]["same_video_pairs"] for c in cats)
    print("total cross-video positive pairs", total_cross, "same-video", total_same)

    # raw crop audit
    train = json.loads((ROOT / "data" / "raw" / "tao" / "annotations" / "train.json").read_text())
    img_map = {im["id"]: im for im in train["images"]}
    missing = [r for r in rows if r["image_id"] not in img_map]
    import os
    frames_root = ROOT / "data" / "raw" / "tao" / "frames"
    missing_file = 0
    for r in rows:
        fn = img_map.get(r["image_id"], {}).get("file_name")
        if fn is None or not (frames_root / fn).exists():
            missing_file += 1
    feats = np.load(ROOT / "outputs" / "iclr27_phase4t" / "train_stream" / "feats.npz")["feats"]
    print("missing image_id:", len(missing), "missing frame file:", missing_file,
          "feats shape", feats.shape, "aligned", feats.shape[0] == len(rows))

    out = ROOT / "outputs" / "iclr27_phase4u" / "data_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "known_categories": len(cats),
        "per_category": {str(c): v for c, v in capacity.items()},
        "raw_crops": {
            "status": "RAW_TRACK_CROPS_AVAILABLE" if missing_file == 0 and len(missing) == 0 else "PARTIAL",
            "rows": len(rows), "missing_image_id": len(missing), "missing_file": missing_file,
        },
    }, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
