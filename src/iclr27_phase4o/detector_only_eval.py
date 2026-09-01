"""Phase 4O detector-only benchmark (unified IoU>=0.5 protocol).

Two input modes:
  --labeled-csv: rows already carrying gt_role (used for D0 frozen
                 stream, from the Phase 4N detection population);
  --proposals-csv: raw proposals (video_id, frame_id, bbox_xyxy,
                   score) matched to TAO GT here.

Outputs summary, novel-recall-FP curve, fixed-FP and fixed-novel-recall
comparisons, and TopK proposal-budget recall.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / \
    "supported_known_ids.json"
TAO = {
    "dev": ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
    "tao_subset" / "validation_20.json",
    "heldout": ROOT / "outputs" / "iclr27_phase4n" / "audit" /
    "validation_heldout_tao_corrected.json",
}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / ua if ua > 0 else 0.0


def load_gt(tao_json):
    known = set(json.loads(KNOWN_IDS.read_text()))
    d = json.loads(tao_json.read_text())
    out = {}
    for a in d["annotations"]:
        if a.get("iscrowd"):
            continue
        b = a["bbox"]
        out.setdefault(a["image_id"], []).append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "category_id": int(a["category_id"]),
            "role": "known" if int(a["category_id"]) in known else "novel",
        })
    return out, {im["id"]: im["video_id"] for im in d["images"]}


def label_proposals(rows, gt, img_vid):
    """Match raw proposals to GT; mark unmatched as fp."""
    for r in rows:
        image_id = int(r["image_id"])
        bbox = [float(v) for v in json.loads(r["bbox_xyxy"])]
        best, bi = None, 0.5
        for g in gt.get(image_id, []):
            v = iou(bbox, g["bbox"])
            if v >= bi:
                bi, best = v, g
        r["gt_role"] = (best or {}).get("role", "fp")
    return rows


def eval_rows(rows, n_frames, gt_counts):
    scores = np.asarray([float(r["score"]) for r in rows])
    roles = np.asarray([r["gt_role"] for r in rows])
    order = np.argsort(-scores)
    n_novel = gt_counts["novel"]
    n_known = gt_counts["known"]
    n_valid = n_novel + n_known
    n_fp = int((roles == "fp").sum())
    cum_novel = np.cumsum(roles[order] == "novel")
    cum_known = np.cumsum(roles[order] == "known")
    cum_fp = np.cumsum(roles[order] == "fp")
    cum_valid = cum_novel + cum_known
    total = np.arange(1, len(rows) + 1)
    novel_recall = cum_novel / max(n_novel, 1)
    known_recall = cum_known / max(n_known, 1)
    valid_recall = cum_valid / max(n_valid, 1)
    precision = cum_valid / total
    fp_per_frame = cum_fp / max(n_frames, 1)
    return {
        "order": order, "scores": scores[order], "roles": roles[order],
        "novel_recall": novel_recall, "known_recall": known_recall,
        "valid_recall": valid_recall, "precision": precision,
        "fp_per_frame": fp_per_frame, "n_novel": n_novel,
        "n_known": n_known, "n_fp": n_fp, "n_rows": len(rows),
    }


def interp_fp_at_recall(recall, fp, target):
    """FP/frame at a given recall (last point reaching it)."""
    idx = np.where(recall >= target)[0]
    if len(idx) == 0:
        return None
    return float(fp[idx[0]])


def interp_recall_at_fp(fp, recall, target):
    idx = np.where(fp <= target)[0]
    if len(idx) == 0:
        return None
    return float(recall[idx[-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    ap.add_argument("--labeled-csv", type=Path, default=None)
    ap.add_argument("--proposals-csv", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT /
                    "outputs" / "iclr27_phase4o" / "detector_only")
    args = ap.parse_args()
    assert (args.labeled_csv is None) != (args.proposals_csv is None)
    if args.labeled_csv is not None:
        rows = list(csv.DictReader(open(args.labeled_csv)))
    else:
        gt, img_vid = load_gt(TAO[args.mode])
        rows = list(csv.DictReader(open(args.proposals_csv)))
        rows = label_proposals(rows, gt, img_vid)
    # exact frame count
    if args.mode == "dev":
        n_frames = 0
        for p in (ROOT / "outputs" / "iclr27_phase3a" / "smoke" /
                  "replay_packages").iterdir():
            n_frames += len(list(p.glob("frame_*.npz")))
    else:
        n_frames = len({im["id"] for im in json.loads(
            TAO["heldout"].read_text())["images"]})
    gt_counts = {
        "known": sum(1 for r in rows if r["gt_role"] == "known"),
        "novel": sum(1 for r in rows if r["gt_role"] == "novel"),
    }
    ev = eval_rows(rows, n_frames, gt_counts)

    # summary operating points at score percentiles
    summary_rows = []
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    n = len(ev["order"])
    for pct in pcts:
        i = min(n - 1, int(n * pct / 100))
        summary_rows.append({
            "detector": args.name, "mode": args.mode,
            "score_percentile": pct,
            "score": round(float(ev["scores"][i]), 4),
            "proposals": i + 1,
            "known_recall": round(float(ev["known_recall"][i]), 4),
            "novel_recall": round(float(ev["novel_recall"][i]), 4),
            "valid_precision": round(float(ev["precision"][i]), 4),
            "fp_per_frame": round(float(ev["fp_per_frame"][i]), 4),
        })
    # full curve on a denser grid for the pareto CSV
    curve = []
    for i in range(0, n, max(1, n // 400)):
        curve.append({
            "detector": args.name, "mode": args.mode,
            "score": round(float(ev["scores"][i]), 4),
            "known_recall": round(float(ev["known_recall"][i]), 4),
            "novel_recall": round(float(ev["novel_recall"][i]), 4),
            "fp_per_frame": round(float(ev["fp_per_frame"][i]), 4),
            "precision": round(float(ev["precision"][i]), 4),
        })
    if curve and curve[-1]["score"] != float(ev["scores"][-1]):
        i = n - 1
        curve.append({
            "detector": args.name, "mode": args.mode,
            "score": round(float(ev["scores"][i]), 4),
            "known_recall": round(float(ev["known_recall"][i]), 4),
            "novel_recall": round(float(ev["novel_recall"][i]), 4),
            "fp_per_frame": round(float(ev["fp_per_frame"][i]), 4),
            "precision": round(float(ev["precision"][i]), 4),
        })
    # fixed FP and fixed novel recall
    fixed_fp = []
    for target in (1.0, 3.0, 5.0, 10.0):
        nr = interp_recall_at_fp(ev["fp_per_frame"],
                                 ev["novel_recall"], target)
        fixed_fp.append({
            "detector": args.name, "mode": args.mode,
            "fp_per_frame": target,
            "novel_recall": round(nr, 4) if nr is not None else "",
            "known_recall": round(float(ev["known_recall"][
                np.where(ev["fp_per_frame"] <= target)[0][-1]]), 4)
            if np.any(ev["fp_per_frame"] <= target) else "",
        })
    fixed_nr = []
    for target in (0.3, 0.5, 0.7):
        fp = interp_fp_at_recall(ev["novel_recall"], ev["fp_per_frame"],
                                 target)
        fixed_nr.append({
            "detector": args.name, "mode": args.mode,
            "novel_recall": target,
            "fp_per_frame": round(fp, 4) if fp is not None else "",
        })
    # TopK budget
    budget = []
    for k in (10, 30, 100):
        if k > n:
            continue
        budget.append({
            "detector": args.name, "mode": args.mode, "top_k": k,
            "known_recall": round(float(ev["known_recall"][k - 1]), 4),
            "novel_recall": round(float(ev["novel_recall"][k - 1]), 4),
            "valid_precision": round(float(ev["precision"][k - 1]), 4),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(name, rows):
        if not rows:
            return
        with open(args.out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(f"{args.name}_{args.mode}_summary.csv", summary_rows)
    def append(name, rows):
        if not rows:
            return
        p = args.out_dir / name
        new = not p.exists()
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)

    append("novel_recall_fp_curve.csv", curve)
    append("fixed_fp_comparison.csv", fixed_fp)
    append("fixed_novel_recall.csv", fixed_nr)
    append("proposal_budget.csv", budget)
    write(f"{args.name}.csv", [{
        "detector": args.name, "mode": args.mode,
        "n_proposals": ev["n_rows"], "n_known_gt": ev["n_known"],
        "n_novel_gt": ev["n_novel"], "n_fp": ev["n_fp"],
        "fp_per_frame_all": round(ev["n_fp"] / max(n_frames, 1), 4),
        "novel_recall_all": 1.0, "known_recall_all": 1.0,
    }])
    print("DETECTOR_ONLY_DONE", args.name, args.mode, "proposals",
          ev["n_rows"], "frames", n_frames)


if __name__ == "__main__":
    main()
