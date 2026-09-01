"""Build forensic counterfactual streams and replay frozen Phase 5A.

S0: current Phase 5A full stream (as-is).
S1: official final-output only (identical to S0 by construction, because
    proposals_dev.csv is already the OVTR tao_track.json filtered to the
    dev annotated frames; kept for completeness).
S2: category-agnostic geometry-aligned diagnostic (rows of tracks with at
    least one frame IoU>=0.5 to any GT box). ORACLE_DIAGNOSTIC_ONLY.
S3: GT-independent deduplicated stream (per frame, greedy IoU>=0.5
    suppression by score).
S4: GT-independent fragment-normalization diagnostic (union-find merge of
    same-video tracks that overlap any frame at IoU>=0.5). IDs change; this
    is diagnostic only and never a main-protocol operation.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def write_stream(out_prefix, rows, feats):
    # recompute causal prior_hits and proposal_local_id
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(int(r["video_id"]), int(r["track_id"]))].append(i)
    for k in by_track:
        by_track[k].sort(key=lambda i: (int(rows[i]["frame_id"]),
                                        int(rows[i].get("proposal_local_id") or 0)))
    prior = {}
    for k, idxs in by_track.items():
        seen = set()
        for i in idxs:
            fr = int(rows[i]["frame_id"])
            prior[i] = len(seen)
            seen.add(fr)
    per_frame = defaultdict(list)
    for i, r in enumerate(rows):
        per_frame[(int(r["video_id"]), int(r["frame_id"]))].append(i)
    for k, idxs in per_frame.items():
        idxs.sort(key=lambda i: (int(rows[i]["track_id"]), -float(rows[i]["score"])))
        for j, i in enumerate(idxs):
            rows[i]["proposal_local_id"] = j
            rows[i]["prior_hits"] = prior[i]
    fieldnames = ["video_id", "frame_id", "image_id", "proposal_local_id",
                  "track_id", "score", "bbox_xyxy", "category_id",
                  "gt_role", "gt_iou", "gt_category_id", "prior_hits"]
    with open(f"{out_prefix}_proposals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    np.savez_compressed(f"{out_prefix}_feats.npz", feats=np.asarray(feats,
                                                                    dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--feats", default="outputs/iclr27_phase4s/q1_features/feats.npz")
    ap.add_argument("--forensic", default="outputs/iclr27_phase5b/audit/geometry/track_forensic_table.csv")
    ap.add_argument("--out", default="outputs/iclr27_phase5b/counterfactual")
    args = ap.parse_args()

    all_rows = load_proposals(Path(args.csv))
    arr = np.load(ROOT / args.feats)["feats"]
    assert len(arr) == len(all_rows)
    tracks = group_tracks(all_rows)
    stream, labels_all = load_gt_tracks_dev()
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # S0 (copy)
    s0 = [dict(r) for r in all_rows]
    write_stream(out / "S0", s0, arr)

    # S2: rows of tracks with >=1 frame IoU>=0.5 to GT
    geom_ok = set()
    with open(args.forensic) as f:
        for r in csv.DictReader(f):
            if int(r["matched_frames_0.5"]) > 0:
                geom_ok.add((int(r["video_id"]), int(r["track_id"])))
    keep = [i for i, r in enumerate(all_rows)
            if (int(r["video_id"]), int(r["track_id"])) in geom_ok]
    write_stream(out / "S2_geom_aligned_oracle", [dict(all_rows[i]) for i in keep],
                 arr[keep])

    # S3: per-frame greedy dedup by score (IoU>=0.5 suppression)
    by_frame = defaultdict(list)
    for i, r in enumerate(all_rows):
        by_frame[(int(r["video_id"]), int(r["frame_id"]))].append(i)
    keep3 = []
    for k, idxs in by_frame.items():
        idxs.sort(key=lambda i: (-float(all_rows[i]["score"]),
                                 int(all_rows[i]["track_id"])))
        kept = []
        for i in idxs:
            b = json.loads(all_rows[i]["bbox_xyxy"])
            if all(box_iou(b, json.loads(all_rows[j]["bbox_xyxy"])) < 0.5
                   for j in kept):
                kept.append(i)
        keep3.extend(kept)
    keep3.sort()
    write_stream(out / "S3_dedup", [dict(all_rows[i]) for i in keep3], arr[keep3])

    # S4: union-find merge of same-video tracks with any-frame IoU>=0.5
    n = len(all_rows)
    parent = list(range(len(tracks)))
    key_index = {k: i for i, k in enumerate(tracks)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    per_frame_tracks = defaultdict(list)
    for i, r in enumerate(all_rows):
        per_frame_tracks[(int(r["video_id"]), int(r["frame_id"]))].append(i)
    for k, idxs in per_frame_tracks.items():
        boxes = [(i, json.loads(all_rows[i]["bbox_xyxy"])) for i in idxs]
        for a in range(len(boxes)):
            for b in range(a + 1, len(boxes)):
                if box_iou(boxes[a][1], boxes[b][1]) >= 0.5:
                    ka = (int(all_rows[boxes[a][0]]["video_id"]),
                          int(all_rows[boxes[a][0]]["track_id"]))
                    kb = (int(all_rows[boxes[b][0]]["video_id"]),
                          int(all_rows[boxes[b][0]]["track_id"]))
                    union(key_index[ka], key_index[kb])
    comp = {}
    for k in tracks:
        comp.setdefault(find(key_index[k]), []).append(k)
    new_id = {}
    next_tid = 0
    for root, keys in comp.items():
        for k in keys:
            new_id[k] = next_tid
        next_tid += 1
    s4 = []
    for r in all_rows:
        rr = dict(r)
        rr["track_id"] = new_id[(int(r["video_id"]), int(r["track_id"]))]
        s4.append(rr)
    write_stream(out / "S4_frag_norm_diag", s4, arr)

    meta = {
        "S0_rows": len(all_rows), "S0_tracks": len(tracks),
        "S2_rows": len(keep), "S2_tracks": len(geom_ok),
        "S3_rows": len(keep3),
        "S3_tracks": len({(int(all_rows[i]["video_id"]),
                           int(all_rows[i]["track_id"])) for i in keep3}),
        "S4_rows": len(s4), "S4_tracks": len(comp),
        "note": "S2/S4 are ORACLE_DIAGNOSTIC_ONLY; S3 is a GT-independent "
                "interface-level dedup; S1==S0 by construction.",
    }
    (out / "stream_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
