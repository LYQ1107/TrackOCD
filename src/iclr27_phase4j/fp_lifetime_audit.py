"""FP lifetime audit: are FP tracklets short-lived?

Uses B0 predictions (physical tracks) + GT overlap to label FP tracklets,
then attaches frame-online semantic observations (Phase 4I B2 l=0.1 logs)
to report length distribution, detector score, p_known / best_novel,
semantic stability, and novel-memory contribution of FP tracklets.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt():
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        b = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
        })
    return out


def load_sem(log_root):
    out = defaultdict(list)
    if log_root is None:
        return out
    for p in log_root.glob("*.jsonl"):
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r.get("physical_track_id")].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--sem-log-root", type=Path, default=None)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()
    gt = load_gt()
    sem = load_sem(args.sem_log_root)

    tracks = defaultdict(list)
    for p in args.pred_dir.glob("*.json"):
        image_id = int(p.stem)
        for r in json.loads(p.read_text()):
            b = r["bbox"]
            tracks[r["track_id"]].append({
                "image_id": image_id,
                "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
                "score": float(r["score"]),
            })

    rows = []
    for tid, dets in tracks.items():
        dets.sort(key=lambda d: d["image_id"])
        matched = sum(1 for d in dets if any(
            iou(d["bbox"], g["bbox"]) >= 0.5 for g in gt.get(d["image_id"], [])))
        is_fp = matched == 0
        sems = sem.get(tid, [])
        sems.sort(key=lambda r: (r["frame_id"], r["det_idx"]))
        p_known = np.mean([r["p_known"] for r in sems]) if sems else float("nan")
        best_novel = np.mean([r["best_novel"] for r in sems]) if sems else float("nan")
        novel_obs = sum(1 for r in sems if r["p_known"] < 0.5) if sems else 0
        novel_ids = {r.get("novel_id") for r in sems if r.get("novel_id")}
        rows.append({
            "track_id": tid, "length": len(dets), "is_fp": int(is_fp),
            "mean_score": float(np.mean([d["score"] for d in dets])),
            "sem_rows": len(sems),
            "mean_p_known": float(p_known) if sems else "",
            "mean_best_novel": float(best_novel) if sems else "",
            "novel_observation_count": novel_obs,
            "distinct_novel_ids": len(novel_ids),
            "max_gap": max([int(b["image_id"]) - int(a["image_id"])
                            for a, b in zip(dets, dets[1:])], default=0),
        })
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fps = [r for r in rows if r["is_fp"]]
    n = len(fps)
    if n:
        for L in (1, 2, 3, 5):
            share = sum(1 for r in fps if r["length"] <= L) / n
            print(f"FP length <= {L}: {share:.4f}  ({sum(1 for r in fps if r['length'] <= L)}/{n})")
        print("FP with semantic rows:", sum(1 for r in fps if r["sem_rows"] > 0))
        print("FP mean length:", float(np.mean([r['length'] for r in fps])))
        print("FP mean novel observation count:",
              float(np.mean([r['novel_observation_count'] for r in fps])))
        print("FP with >=1 distinct novel id:",
              sum(1 for r in fps if r["distinct_novel_ids"] > 0))


if __name__ == "__main__":
    main()
