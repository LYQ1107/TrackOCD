#!/usr/bin/env python3
"""Mechanism audit for the DSCQ physical/semantic states.

Joins the eval-time per-proposal DSCQ state dump with the frozen
dev/heldout proposal protocol and reports:
  - E-state separation (persistent valid vs persistent FP)
  - birth evidence (new valid vs new FP)
  - semantic reliability by group
  - known/novel routing logits by role
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_VAL = ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json"
DEV_GT = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
HO_GT = ROOT / "outputs" / "iclr27_phase4n" / "audit" / "validation_heldout_tao_corrected.json"


def load_proposals(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "video_id": int(r["video_id"]),
                "frame_id": int(r["frame_id"]),
                "track_id": int(r["track_id"]),
                "gt_role": r["gt_role"],
                "prior_hits": int(r["prior_hits"]),
            })
    return rows


def build_lookup(proposals):
    lut = {}
    for r in proposals:
        lut[(r["video_id"], r["frame_id"], r["track_id"])] = r
    return lut


def map_file_to_video(tao_val):
    mapping = {}
    for v in tao_val["videos"]:
        mapping[v["name"]] = int(v["id"])
    return mapping


def map_frames(tao_val):
    frames = {}
    for im in tao_val["images"]:
        frames[(int(im["video_id"]), int(im["frame_index"]))] = int(im["id"])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dscq-stats", required=True)
    ap.add_argument("--dev-csv", required=True)
    ap.add_argument("--heldout-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    stats = json.loads(Path(args.dscq_stats).read_text())
    dev_lut = build_lookup(load_proposals(args.dev_csv))
    ho_lut = build_lookup(load_proposals(args.heldout_csv))
    tao = json.loads(TAO_VAL.read_text())
    vid_map = map_file_to_video(tao)
    frame_map = map_frames(tao)

    groups = defaultdict(list)
    for s in stats:
        fp = s["file_path"]
        parts = fp.split("/")
        vname = "/".join(parts[:2]) if len(parts) >= 2 else fp
        if vname not in vid_map:
            continue
        vid = vid_map[vname]
        fid = int(s["frame_id"])
        if (vid, fid) not in frame_map:
            continue
        tid = int(s["obj_idx"])
        row = dev_lut.get((vid, fid, tid)) or ho_lut.get((vid, fid, tid))
        if row is None:
            continue
        role = row["gt_role"]
        hit = int(row["prior_hits"])
        if role in ("known", "novel"):
            group = f"persistent_valid" if hit >= 1 else "new_valid"
        else:
            group = "persistent_fp" if hit >= 1 else "new_fp"
        groups[group].append({
            "e_valid": float(s["e_valid_logit"]),
            "birth": float(s["birth_logit"]),
            "e_rel": float(s["e_reliability"]),
            "s_rel": float(s["s_reliability"]),
            "s_known": float(s["s_known_logit"]),
            "s_novel": float(s["s_novel_logit"]),
        })

    report = {}
    for g, vals in groups.items():
        arr = np.array([[v["e_valid"], v["birth"], v["e_rel"], v["s_rel"],
                         v["s_known"], v["s_novel"]] for v in vals])
        report[g] = {
            "n": len(vals),
            "e_valid_logit_mean": float(arr[:, 0].mean()),
            "birth_logit_mean": float(arr[:, 1].mean()),
            "e_reliability_mean": float(arr[:, 2].mean()),
            "s_reliability_mean": float(arr[:, 3].mean()),
            "s_known_minus_novel_mean": float((arr[:, 4] - arr[:, 5]).mean()),
        }

    sep = None
    if "persistent_valid" in report and "persistent_fp" in report:
        sep = report["persistent_valid"]["e_valid_logit_mean"] - \
            report["persistent_fp"]["e_valid_logit_mean"]
    birth_sep = None
    if "new_valid" in report and "new_fp" in report:
        birth_sep = report["new_valid"]["birth_logit_mean"] - \
            report["new_fp"]["birth_logit_mean"]
    report["summary"] = {
        "e_valid_separation_persistent": sep,
        "birth_logit_separation_new": birth_sep,
        "s_reliability_separation": (
            report["persistent_valid"]["s_reliability_mean"] -
            report["persistent_fp"]["s_reliability_mean"]
            if "persistent_valid" in report and "persistent_fp" in report
            else None),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("DSCQ_MECHANISM_AUDIT_DONE")


if __name__ == "__main__":
    main()
