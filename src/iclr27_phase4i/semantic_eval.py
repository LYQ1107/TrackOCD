"""Frame-online semantic diagnostics on the 20-video subset.

Grounds every predicted detection to TAO GT by IoU >= 0.5 (offline
evaluation only; GT never enters the online loop) and reports:
  - per-frame routing accuracy (known/novel) on GT-matched detections;
  - Known->Novel / Novel->Known frame-level rates;
  - known-class accuracy among GT-known detections;
  - semantic-ID switches along physical tracks;
  - novel semantic reuse consistency per GT novel track;
  - false novel semantic birth rate;
  - Association-Correct Semantic Accuracy.
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


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(known_ids):
    d = json.loads(TAO_JSON.read_text())
    out = defaultdict(list)
    for ann in d["annotations"]:
        cat = int(ann["category_id"])
        bbox = ann["bbox"]
        out[ann["image_id"]].append({
            "bbox": [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]],
            "track_id": int(ann["track_id"]),
            "category_id": cat,
            "role": "known" if cat in known_ids else "novel",
        })
    return out


def evaluate(log_root, out_csv):
    known = set(json.loads(KNOWN_IDS.read_text()))
    gt = load_gt(known)
    per_video = {}
    novel_id_support = defaultdict(int)
    track_sem = defaultdict(list)
    gt_novel_track_ids = defaultdict(set)
    gt_track_pred_ids = defaultdict(list)

    for log_path in sorted(log_root.glob("*.jsonl")):
        vid = int(log_path.stem)
        stats = defaultdict(int)
        matched = 0
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid = r["physical_track_id"]
            if tid is not None:
                track_sem[tid].append(r)
            sem_id = r.get("novel_id")
            if sem_id is not None:
                novel_id_support[sem_id] += 1
            pred_role = "known" if r["p_known"] >= 0.5 else "novel"
            cands = gt.get(int(r["image_id"]), [])
            best = None
            best_iou = 0.5
            for g in cands:
                v = iou(r["bbox"], g["bbox"])
                if v >= best_iou:
                    best_iou = v
                    best = g
            if best is None:
                stats["unmatched"] += 1
                continue
            matched += 1
            stats["matched"] += 1
            if pred_role == best["role"]:
                stats["routing_correct"] += 1
            if best["role"] == "known" and pred_role == "novel":
                stats["k2n"] += 1
            if best["role"] == "novel" and pred_role == "known":
                stats["n2k"] += 1
            if best["role"] == "known":
                stats["known_matched"] += 1
            else:
                stats["novel_matched"] += 1
            if best["role"] == "known" and pred_role == "known":
                if r.get("known_class_id") == best["category_id"]:
                    stats["known_class_correct"] += 1
                stats["known_class_total"] += 1
            if tid is not None:
                gt_track_pred_ids[best["track_id"]].append(tid)
            if best["role"] == "novel":
                sem_id = r.get("novel_id") if pred_role == "novel" else None
                gt_novel_track_ids[best["track_id"]].add(
                    sem_id if sem_id is not None else -1)
        per_video[vid] = stats

    # aggregate
    agg = defaultdict(int)
    for s in per_video.values():
        for k, v in s.items():
            agg[k] += v
    sem_switches = 0
    sem_switch_total = 0
    for tid, seq in track_sem.items():
        seq = sorted(seq, key=lambda r: (r["frame_id"], r["det_idx"]))
        prev = None
        for r in seq:
            cur = r.get("semantic_id")
            if prev is not None and cur != prev:
                sem_switches += 1
            if prev is not None:
                sem_switch_total += 1
            prev = cur
    novel_consistent = sum(1 for ids in gt_novel_track_ids.values()
                           if len(ids) == 1 and -1 not in ids)
    novel_total = len(gt_novel_track_ids)
    false_births = sum(1 for v in novel_id_support.values() if v == 1)

    from collections import Counter
    gt_majority = {gt_tid: Counter(pids).most_common(1)[0][0]
                   for gt_tid, pids in gt_track_pred_ids.items()}
    assoc_correct = 0
    assoc_total = 0
    for log_path in sorted(log_root.glob("*.jsonl")):
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cands = gt.get(int(r["image_id"]), [])
            best, bi = None, 0.5
            for g in cands:
                v = iou(r["bbox"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            if best is None or r.get("physical_track_id") is None:
                continue
            assoc_total += 1
            if r["physical_track_id"] == gt_majority.get(best["track_id"]):
                assoc_correct += 1

    summary = {
        "routing_accuracy": agg["routing_correct"] / max(agg["matched"], 1),
        "k2n_rate_known_denom": agg["k2n"] /
            max(agg["known_matched"], 1),
        "n2k_rate_novel_denom": agg["n2k"] /
            max(agg["novel_matched"], 1),
        "known_class_accuracy": agg["known_class_correct"] /
            max(agg["known_class_total"], 1),
        "semantic_id_switch_rate": sem_switches /
            max(sem_switch_total, 1),
        "novel_semantic_consistent_tracks": novel_consistent,
        "novel_gt_tracks": novel_total,
        "novel_consistency": novel_consistent / max(novel_total, 1),
        "novel_semantic_ids": len(novel_id_support),
        "false_semantic_births": false_births,
        "assoc_correct_rate": assoc_correct / max(assoc_total, 1),
        "matched_detections": agg["matched"],
        "unmatched_detections": agg["unmatched"],
    }
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    evaluate(args.log_root, args.out)


if __name__ == "__main__":
    main()
