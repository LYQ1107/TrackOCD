"""Phase 4W category capacity audit + category-disjoint meta split.

Only 48 supported-known TRAIN categories are used. The split is
TRAIN-only; dev / true-novel / DIAGNOSTIC labels are never touched.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from src.iclr27_phase4t.stream_data import build_tracklets
from src.iclr27_phase4u.data import ROOT


def load_store():
    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np_load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    return rows, feats


def np_load(p):
    import numpy as np
    return np.load(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase4w/meta_split/capacity.json")
    ap.add_argument("--by-cat-out", default="outputs/iclr27_phase4w/meta_split/by_cat.json")
    ap.add_argument("--n-meta-dev", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260815)
    args = ap.parse_args()

    rows, _ = load_store()
    tracklets = build_tracklets(rows)
    by_cat: dict[int, list[list]] = defaultdict(list)
    for key, tl in tracklets.items():
        if tl["role"] != "known":
            continue
        by_cat[tl["gt_category_id"]].append([key[0], key[1], tl["length"]])

    stats = {}
    for c in sorted(by_cat):
        keys = by_cat[c]
        phys = set()
        vids = set()
        for k in keys:
            for r in tracklets[(k[0], k[1])]["rows"]:
                if r["gt_role"] != "fp":
                    phys.add((r["video_id"], r["gt_track_id"]))
                    vids.add(r["video_id"])
        stats[str(c)] = {
            "n_tracklets": len(keys),
            "n_tracklets_len_ge2": sum(1 for k in keys if k[2] >= 2),
            "n_physical_instances": len(phys),
            "n_videos": len(vids),
        }

    # candidate meta-dev categories: >=2 physical instances, >=2 videos,
    # >=4 tracklets with length>=2 (enough for pseudo-novel tracklets)
    cand = [int(c) for c, s in stats.items()
            if s["n_physical_instances"] >= 2 and s["n_videos"] >= 2
            and s["n_tracklets_len_ge2"] >= 4]
    import random
    rng = random.Random(args.seed)
    rng.shuffle(cand)
    meta_dev = sorted(cand[: args.n_meta_dev])
    meta_train = [c for c in sorted(by_cat) if c not in meta_dev]
    # avoid the dominant category (805) in meta-dev so training is not hollow
    if 805 in meta_dev:
        meta_dev = [c for c in meta_dev if c != 805]
        repl = [c for c in cand if c != 805 and c not in meta_dev]
        rng.shuffle(repl)
        if repl:
            meta_dev = sorted(meta_dev + [repl[0]])
            meta_train = [c for c in sorted(by_cat) if c not in meta_dev]

    out = {
        "n_categories_total": len(stats),
        "meta_dev_categories": meta_dev,
        "meta_train_categories": meta_train,
        "n_meta_dev": len(meta_dev),
        "n_meta_train": len(meta_train),
        "capacity_thresholds": {
            "ge2_phys": sum(1 for s in stats.values() if s["n_physical_instances"] >= 2),
            "ge2_videos": sum(1 for s in stats.values() if s["n_videos"] >= 2),
            "ge4_tracklets_len2": sum(1 for s in stats.values()
                                      if s["n_tracklets_len_ge2"] >= 4),
        },
        "per_category": stats,
        "candidate_meta_dev": cand,
    }
    out_dir = ROOT / Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(out, indent=2))
    (ROOT / args.by_cat_out).write_text(json.dumps(
        {str(c): by_cat[c] for c in sorted(by_cat)}, indent=2))
    print(json.dumps({
        "n_meta_dev": len(meta_dev), "meta_dev": meta_dev,
        "n_meta_train": len(meta_train),
        "capacity": out["capacity_thresholds"],
    }, indent=2))


if __name__ == "__main__":
    main()
