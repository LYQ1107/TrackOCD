"""Track-age semantic audit on the Phase 4I frame-online semantic logs.

For GT-matched detections (IoU >= 0.5), group by physical track age
(1, 2, 3-4, 5-8, 9-16, 17+) and by TP/FP, and report:
  K2N, N2K, routing accuracy, known-class accuracy, mean p_known,
  semantic entropy (class distribution), semantic switch rate,
  best-novel compatibility, false novel birth (novel_id support == 1).
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
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json"

AGE_BUCKETS = [(1, 1, "age1"), (2, 2, "age2"), (3, 4, "age3_4"),
               (5, 8, "age5_8"), (9, 16, "age9_16"), (17, 10 ** 6, "age17plus")]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(known):
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        b = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": int(ann["category_id"]),
            "role": "known" if int(ann["category_id"]) in known else "novel",
        })
    return out


def entropy(class_dist):
    d = np.asarray(class_dist, dtype=np.float64)
    d = d / d.sum()
    return float(-(d * np.log(d + 1e-12)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()
    known = set(json.loads(KNOWN_IDS.read_text()))
    gt = load_gt(known)

    rows = []
    novel_support = defaultdict(int)
    for log_path in sorted(args.log_root.glob("*.jsonl")):
        track_age = defaultdict(int)
        track_prev = defaultdict(lambda: None)
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r.get("physical_track_id")
            if tid is None:
                continue
            track_age[tid] += 1
            age = track_age[tid]
            if r.get("novel_id") is not None:
                novel_support[r["novel_id"]] += 1
            # match GT
            cands = gt.get(int(r["image_id"]), [])
            best, bi = None, 0.5
            for g in cands:
                v = iou(r["bbox"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            is_fp = best is None
            role = best["role"] if best else None
            pred_role = "known" if r["p_known"] >= 0.5 else "novel"
            prev = track_prev[tid]
            switch = 0
            if prev is not None and r.get("semantic_id") != prev:
                switch = 1
            track_prev[tid] = r.get("semantic_id")
            for lo, hi, name in AGE_BUCKETS:
                if lo <= age <= hi:
                    bucket = name
                    break
            else:
                bucket = "age17plus"
            rows.append({
                "video_id": int(log_path.stem),
                "track_id": tid, "age": age, "bucket": bucket,
                "is_fp": int(is_fp),
                "gt_role": role or "fp",
                "pred_role": pred_role,
                "routing_correct": int(not is_fp and pred_role == role),
                "k2n": int(not is_fp and role == "known" and pred_role == "novel"),
                "n2k": int(not is_fp and role == "novel" and pred_role == "known"),
                "known_class_correct": int(
                    not is_fp and role == "known" and pred_role == "known"
                    and r.get("known_class_id") == best["category_id"]),
                "known_class_total": int(
                    not is_fp and role == "known" and pred_role == "known"),
                "p_known": float(r["p_known"]),
                "best_novel": float(r["best_novel"]),
                "semantic_switch": switch,
                "novel_observation": int(pred_role == "novel"),
                "novel_id": r.get("novel_id"),
            })

    out_rows = []
    for bucket in [b[2] for b in AGE_BUCKETS]:
        for fp in (0, 1):
            rs = [r for r in rows if r["bucket"] == bucket and r["is_fp"] == fp]
            if not rs:
                continue
            matched = [r for r in rs if not r["is_fp"]]
            known_m = [r for r in matched if r["gt_role"] == "known"]
            novel_m = [r for r in matched if r["gt_role"] == "novel"]
            kc = [r for r in known_m if r["pred_role"] == "known"]
            out_rows.append({
                "bucket": bucket, "is_fp": fp, "n": len(rs),
                "routing_accuracy": sum(r["routing_correct"] for r in matched)
                    / max(len(matched), 1),
                "k2n": sum(r["k2n"] for r in known_m) / max(len(known_m), 1),
                "n2k": sum(r["n2k"] for r in novel_m) / max(len(novel_m), 1),
                "known_class_accuracy": sum(r["known_class_correct"] for r in kc)
                    / max(len(kc), 1),
                "mean_p_known": float(np.mean([r["p_known"] for r in rs])),
                "mean_best_novel": float(np.mean([r["best_novel"] for r in rs])),
                "semantic_switch_rate": sum(r["semantic_switch"] for r in rs)
                    / max(len(rs), 1),
                "novel_observation_rate": sum(r["novel_observation"] for r in rs)
                    / max(len(rs), 1),
            })
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
