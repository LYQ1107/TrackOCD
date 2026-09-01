#!/usr/bin/env python3
"""Evaluate OVTR P0/P1/P2 tracking JSON under the frozen dev/heldout
proposal protocol (same matching and FP/frame conventions as Phase 4O)."""
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
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json"

AGE_BUCKETS = [
    ("age0", 0, 0),
    ("age1", 1, 1),
    ("age2", 2, 2),
    ("age3_4", 3, 4),
    ("age5plus", 5, 10 ** 9),
]


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


def load_gt(path, known_ids):
    d = json.loads(path.read_text())
    gt = {}
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        b = a["bbox"]
        gt.setdefault(int(a["image_id"]), []).append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "role": "known" if int(a["category_id"]) in known_ids else "novel",
            "category_id": int(a["category_id"]),
        })
    return gt, {im["id"] for im in d["images"]}, len(d["images"])


def build_rows(results, mode):
    known_ids = set(json.loads(KNOWN_IDS.read_text()))
    gt_path = DEV_GT if mode == "dev" else HO_GT
    gt, gt_images, n_frames = load_gt(gt_path, known_ids)
    val = json.loads(TAO_VAL.read_text())
    img_map = {im["id"]: im for im in val["images"]}

    per_frame = defaultdict(list)
    for r in results:
        img_id = int(r["image_id"])
        if img_id not in gt_images:
            continue
        per_frame[img_id].append(r)
    for img_id in per_frame:
        per_frame[img_id].sort(
            key=lambda r: (int(r.get("track_id", -1)),
                           float(r.get("score", 0.0))))
        for i, r in enumerate(per_frame[img_id]):
            r["_pid"] = i

    rows = []
    for r in results:
        img_id = int(r["image_id"])
        if img_id not in gt_images:
            continue
        im = img_map[img_id]
        bb = [float(v) for v in r["bbox"][:4]]
        bbox = [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]]
        best, role, best_cat = 0.5, "fp", -1
        for g in gt.get(img_id, []):
            v = iou(bbox, g["bbox"])
            if v >= best:
                best, role, best_cat = v, g["role"], g["category_id"]
        rows.append({
            "video_id": int(r["video_id"]),
            "frame_id": int(im["frame_index"]),
            "image_id": img_id,
            "proposal_local_id": r["_pid"],
            "track_id": int(r.get("track_id", -1)),
            "score": float(r.get("score", 0.0)),
            "bbox_xyxy": json.dumps(bbox),
            "category_id": int(r.get("category_id", -1)),
            "sem_action": r.get("sem_action", ""),
            "sem_sid": r.get("sem_sid", ""),
            "gt_role": role,
            "gt_iou": float(best),
            "gt_category_id": best_cat,
        })
    rows.sort(key=lambda r: (r["video_id"], r["frame_id"],
                             r["proposal_local_id"]))
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(r["video_id"], r["track_id"])].append(i)
    for k in by_track:
        by_track[k].sort(key=lambda i: (rows[i]["frame_id"],
                                        rows[i]["proposal_local_id"]))
    for i, r in enumerate(rows):
        key = (r["video_id"], r["track_id"])
        hist = [j for j in by_track[key] if rows[j]["frame_id"] < r["frame_id"]]
        r["prior_hits"] = len({rows[j]["frame_id"] for j in hist})
    return rows, n_frames


def curve_metrics(rows, n_frames):
    order = np.argsort(-np.asarray([r["score"] for r in rows]),
                       kind="mergesort")
    roles = np.asarray([r["gt_role"] for r in rows])
    ph = np.asarray([r["prior_hits"] for r in rows])
    cum_novel = np.cumsum(roles == "novel")
    cum_known = np.cumsum(roles == "known")
    cum_fp = np.cumsum(roles == "fp")
    cum_pfp = np.cumsum((roles == "fp") & (ph >= 2))
    total_novel = int((roles == "novel").sum())
    total_known = int((roles == "known").sum())
    total_fp = int((roles == "fp").sum())
    total_pfp = int(((roles == "fp") & (ph >= 2)).sum())
    novel_recall = cum_novel / max(total_novel, 1)
    known_recall = cum_known / max(total_known, 1)
    fp_per_frame = cum_fp / max(n_frames, 1)

    def rec_at_fp(target):
        idx = np.where(fp_per_frame <= target)[0]
        return float(novel_recall[idx[-1]]) if len(idx) else 0.0

    def fp_at_recall(target):
        idx = np.where(novel_recall >= target)[0]
        return float(fp_per_frame[idx[0]]) if len(idx) else None

    def cut_at_fp(target):
        idx = np.where(fp_per_frame <= target)[0]
        return int(idx[-1]) if len(idx) else None

    def cut_at_recall(target):
        idx = np.where(novel_recall >= target)[0]
        return int(idx[0]) if len(idx) else None

    def rejection(cut):
        if cut is None:
            return None, None
        all_rej = 1.0 - int(cum_fp[cut]) / total_fp if total_fp else None
        pfp_rej = 1.0 - int(cum_pfp[cut]) / total_pfp if total_pfp else None
        return all_rej, pfp_rej

    age_totals = {}
    age_cum = {}
    for name, lo, hi in AGE_BUCKETS:
        mask = (roles == "novel") & (ph >= lo) & (ph <= hi)
        age_totals[name] = int(mask.sum())
        age_cum[name] = np.cumsum(mask)

    m = {
        "total_rows": len(rows),
        "total_novel": total_novel,
        "total_known": total_known,
        "total_fp": total_fp,
        "total_persistent_fp": total_pfp,
        "persistent_fp_per_frame": total_pfp / max(n_frames, 1),
    }
    for t in (0.1, 0.3, 1.0, 3.0, 5.0):
        cut = cut_at_fp(t)
        m[f"novel_recall_at_fp_{t}"] = rec_at_fp(t)
        m[f"known_recall_at_fp_{t}"] = float(
            known_recall[cut]) if cut is not None else 0.0
        all_rej, pfp_rej = rejection(cut)
        m[f"reject_all_fp_at_fp_{t}"] = all_rej
        m[f"reject_persistent_fp_at_fp_{t}"] = pfp_rej
        for name, _, _ in AGE_BUCKETS:
            m[f"early_{name}_recall_at_fp_{t}"] = (
                float(age_cum[name][cut] / age_totals[name])
                if cut is not None and age_totals[name] else None)
    for t in (0.1, 0.2, 0.3, 0.5):
        m[f"fp_per_frame_at_recall_{t}"] = fp_at_recall(t)
        cut = cut_at_recall(t)
        all_rej, pfp_rej = rejection(cut)
        m[f"reject_all_fp_at_recall_{t}"] = all_rej
        m[f"reject_persistent_fp_at_recall_{t}"] = pfp_rej
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    results = json.loads(Path(args.results_json).read_text())
    out = {}
    for mode in ("dev", "heldout"):
        rows, n_frames = build_rows(results, mode)
        fieldnames = list(rows[0].keys()) if rows else [
            "video_id", "frame_id", "image_id", "proposal_local_id",
            "track_id", "score", "bbox_xyxy", "category_id",
            "sem_action", "sem_sid", "gt_role", "gt_iou",
            "gt_category_id", "prior_hits"]
        csv_path = f"{args.out_prefix}_{mode}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        m = curve_metrics(rows, n_frames)
        m["mode"] = mode
        m["n_frames"] = n_frames
        out[mode] = m
        print(f"{mode}: {len(rows)} rows, "
              f"novel_recall@1FP={m['novel_recall_at_fp_1.0']:.4f}, "
              f"FP/frame@r0.3={m['fp_per_frame_at_recall_0.3']}, "
              f"persistentFP/frame={m['persistent_fp_per_frame']:.4f}")
    with open(f"{args.out_prefix}_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("OVTR_MAIN_EVAL_DONE")


if __name__ == "__main__":
    main()
