"""Join per-track predicted category with official TAO per-video annotation
metadata (neg_category_ids / not_exhaustive_category_ids).

This uses the official TAO validation annotation semantics: a video marks
categories that are verified absent (negative) and categories whose labels
are not guaranteed exhaustive. An unaligned track whose predicted category
is in the video's not-exhaustive list may be a real unannotated object;
one whose predicted category is in the negative list is a strong FP
candidate (the category is verified absent); otherwise coverage is
unverified.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    TAO_VAL_ANN,
    known_ids,
    load_proposals,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--forensic", default="outputs/iclr27_phase5b/audit/geometry/track_forensic_table.csv")
    ap.add_argument("--out", default="outputs/iclr27_phase5b/audit/geometry/track_forensic_table_with_tao.csv")
    args = ap.parse_args()

    val = json.loads(TAO_VAL_ANN.read_text())
    vid_meta = {int(v["id"]): v for v in val["videos"]}
    rows = load_proposals(Path(args.csv))
    pred_cat_by_track = {}
    row_count_by_track = Counter()
    for r in rows:
        key = (int(r["video_id"]), int(r["track_id"]))
        pred_cat_by_track[key] = int(r["category_id"])
        row_count_by_track[key] += 1

    with open(args.forensic) as f:
        table = list(csv.DictReader(f))
    known = known_ids()
    for row in table:
        vid = int(row["video_id"])
        cat = int(row["pred_category"])
        meta = vid_meta.get(vid, {})
        neg = set(meta.get("neg_category_ids") or [])
        ne = set(meta.get("not_exhaustive_category_ids") or [])
        row["video_neg_cats"] = sorted(neg)
        row["video_not_exhaustive_cats"] = sorted(ne)
        row["pred_cat_neg"] = int(cat in neg)
        row["pred_cat_not_exhaustive"] = int(cat in ne)
        if cat in neg:
            status = "category_verified_absent"
        elif cat in ne:
            status = "category_not_exhaustive"
        else:
            status = "coverage_unverified"
        row["annotation_status"] = status

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    # summary: unaligned cause x annotation status x track count and rows
    by = Counter()
    rows_by = Counter()
    for row in table:
        if row["aligned"] == "1":
            continue
        cause = row["forensic_cause"]
        st = row["annotation_status"]
        key = (cause, st)
        by[key] += 1
        # row count for this track
        rows_by[key] += row_count_by_track[
            (int(row["video_id"]), int(row["track_id"]))]
    summary = {
        "unaligned_by_cause_x_status": {f"{c}|{s}": n for (c, s), n in by.items()},
        "unaligned_rows_by_cause_x_status": {f"{c}|{s}": n for (c, s), n in rows_by.items()},
        "status_totals": dict(Counter(r["annotation_status"] for r in table
                                      if r["aligned"] != "1")),
    }
    out = Path(args.out).with_name("annotation_status.json")
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
