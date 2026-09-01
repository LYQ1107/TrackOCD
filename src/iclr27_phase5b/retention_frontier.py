"""Causal physical-stat retention frontier (Phase 5B diagnostic only).

For every physical track, only stats available at the decision frame are
used (first-frame score; optionally first-2-frame mean as a causal
diagnostic). We report:
  - GT track coverage (known and novel separately),
  - unaligned track admission,
  - row-level known/novel recall and FP/frame.

This is not a method and not used to select a semantic model.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--out", default="outputs/iclr27_phase5b/audit/retention")
    args = ap.parse_args()

    rows = load_proposals(Path(args.csv))
    tracks = group_tracks(rows)
    stream, labels_all = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
    gb = gt_track_boxes(stream)
    mapping = align_pred_to_gt(tracks, gb)

    aligned_keys = set(mapping)
    known_keys = [k for k in aligned_keys
                  if labels[mapping[k]]["protocol_role"]
                  in ("supported_known", "zero_shot_known")]
    novel_keys = [k for k in aligned_keys
                  if labels[mapping[k]]["protocol_role"] == "novel"]
    unaligned_keys = [k for k in tracks if k not in aligned_keys]

    track_first = {}
    track_first2 = {}
    for k, tr in tracks.items():
        tr = sorted(tr, key=lambda r: r["frame_id"])
        sc = [r["score"] for r in tr]
        track_first[k] = sc[0]
        track_first2[k] = float(np.mean(sc[:2])) if len(sc) >= 2 else sc[0]

    rows_by_track = defaultdict(list)
    for r in rows:
        rows_by_track[(r["video_id"], r["track_id"])].append(r)
    n_frames = len(set((r["video_id"], r["frame_id"]) for r in rows))
    thresholds = [0.0, 0.1, 0.15, 0.19, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    out_rows = []
    for th in thresholds:
        def admit(keys, score_fn):
            return sum(1 for k in keys if score_fn[k] >= th)
        known_cov = admit(known_keys, track_first) / max(len(known_keys), 1)
        novel_cov = admit(novel_keys, track_first) / max(len(novel_keys), 1)
        unaligned_adm = admit(unaligned_keys, track_first) / max(len(unaligned_keys), 1)
        known_cov2 = admit(known_keys, track_first2) / max(len(known_keys), 1)
        novel_cov2 = admit(novel_keys, track_first2) / max(len(novel_keys), 1)
        unaligned_adm2 = admit(unaligned_keys, track_first2) / max(len(unaligned_keys), 1)
        # row-level
        admitted_rows = [r for k, rs in rows_by_track.items() if track_first[k] >= th
                         for r in rs]
        n_adm = len(admitted_rows)
        known_rows = sum(1 for r in admitted_rows if r["gt_role"] == "known")
        novel_rows = sum(1 for r in admitted_rows if r["gt_role"] == "novel")
        fp_rows = sum(1 for r in admitted_rows if r["gt_role"] == "fp")
        total_known = sum(1 for r in rows if r["gt_role"] == "known")
        total_novel = sum(1 for r in rows if r["gt_role"] == "novel")
        total_fp = sum(1 for r in rows if r["gt_role"] == "fp")
        out_rows.append({
            "threshold": th,
            "known_track_coverage": known_cov,
            "novel_track_coverage": novel_cov,
            "unaligned_track_admission": unaligned_adm,
            "known_track_coverage_first2": known_cov2,
            "novel_track_coverage_first2": novel_cov2,
            "unaligned_track_admission_first2": unaligned_adm2,
            "admitted_rows": n_adm,
            "known_row_recall": known_rows / max(total_known, 1),
            "novel_row_recall": novel_rows / max(total_novel, 1),
            "fp_per_frame": fp_rows / max(n_frames, 1),
            "fp_row_admission": fp_rows / max(total_fp, 1),
        })
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "frontier.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # summary at the OVTR operating threshold 0.19 and 0.5
    summary = {
        "n_known_tracks": len(known_keys),
        "n_novel_tracks": len(novel_keys),
        "n_unaligned_tracks": len(unaligned_keys),
        "operating_0.19": next(r for r in out_rows if r["threshold"] == 0.19),
        "operating_0.5": next(r for r in out_rows if r["threshold"] == 0.5),
    }
    (out / "retention_summary.json").write_text(
        json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
