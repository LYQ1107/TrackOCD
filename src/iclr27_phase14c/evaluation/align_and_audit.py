"""Corrected temporal-IoU alignment and proposal opportunity audit."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def temporal_iou(gt, pred):
    shared = sorted(set(gt) & set(pred))
    return sum(iou(gt[f], pred[f]) for f in shared) / max(len(shared), 1) if shared else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--gt", default="outputs/iclr27_phase14c/manifests/mixed_gt_tracks.jsonl")
    ap.add_argument("--out-csv", default="outputs/iclr27_phase14c/proposals/proposals_aligned.csv")
    ap.add_argument("--out-audit", default="outputs/iclr27_phase14c/eval/proposal_opportunity_audit.json")
    args = ap.parse_args()
    rows = []
    with (ROOT / args.proposals).open() as f:
        for r in csv.DictReader(f):
            r = dict(r)
            for k in ("video_id", "frame_id", "source_frame_index", "image_id", "proposal_local_id", "track_id", "det_category_id", "prior_hits"):
                r[k] = int(r[k])
            r["score"] = float(r["score"])
            r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
            rows.append(r)
    gt = []
    with (ROOT / args.gt).open() as f:
        gt = [json.loads(line) for line in f if line.strip()]
    gt_by_key = {}
    for g in gt:
        g["frame_boxes"] = {int(f): b for f, b in zip(g["frame_indices"], g["boxes_xyxy"])}
        gt_by_key[(int(g["video_id"]), int(g["track_id"]))] = g
    pred_by_key = defaultdict(list)
    for r in rows:
        pred_by_key[(r["video_id"], r["track_id"])].append(r)
    pred_boxes = {k: {int(r["source_frame_index"]): r["bbox_xyxy"] for r in rs} for k, rs in pred_by_key.items()}
    candidates = []
    for pk, pb in pred_boxes.items():
        for gk, g in gt_by_key.items():
            if pk[0] != gk[0]:
                continue
            v = temporal_iou(g["frame_boxes"], pb)
            if v > 0:
                candidates.append((v, pk, gk))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    mapping, used_p, used_g = {}, set(), set()
    for v, pk, gk in candidates:
        if pk in used_p or gk in used_g:
            continue
        used_p.add(pk); used_g.add(gk); mapping[pk] = (gk, v)
    aligned_rows = []
    for r in rows:
        pk = (r["video_id"], r["track_id"])
        if pk in mapping:
            gk, v = mapping[pk]; g = gt_by_key[gk]
            # Labels are added only in this evaluator-side CSV.
            r["gt_track_id"] = g["track_id"]
            r["gt_category_id"] = g["category_id"]
            r["gt_role"] = g["role"]
            r["gt_temporal_iou"] = v
        else:
            r["gt_track_id"] = -1; r["gt_category_id"] = -1; r["gt_role"] = "fp"; r["gt_temporal_iou"] = 0.0
        aligned_rows.append(r)
    out_csv = ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_id", "frame_id", "source_frame_index", "image_id", "proposal_local_id", "track_id", "score", "bbox_xyxy", "det_category_id", "source_family", "prior_hits", "gt_track_id", "gt_category_id", "gt_role", "gt_temporal_iou"]
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in aligned_rows:
            x = dict(r); x["bbox_xyxy"] = json.dumps(x["bbox_xyxy"], separators=(",", ":")); w.writerow(x)
    os.replace(tmp, out_csv)

    aligned_gt = [gt_by_key[gk] for gk, _ in mapping.values()]
    by_cat = Counter(g["category_id"] for g in aligned_gt if g["role"] == "novel")
    novel = [g for g in aligned_gt if g["role"] == "novel"]
    pairs = 0; cross_video = 0
    for i, a in enumerate(novel):
        for b in novel[i + 1:]:
            if a["category_id"] == b["category_id"] and (a["video_id"], a["track_id"]) != (b["video_id"], b["track_id"]):
                pairs += 1
                cross_video += int(a["video_id"] != b["video_id"])
    gt_to_pred = Counter(gk for _, gk in mapping.values())
    audit = {
        "proposal_rows": len(rows),
        "proposal_physical_tracks": len(pred_by_key),
        "proposal_videos": len({r["video_id"] for r in rows}),
        "frames_with_rows": len({(r["video_id"], r["frame_id"]) for r in rows}),
        "aligned_gt_tracks": len(mapping),
        "aligned_known_tracks": sum(g["role"] == "supported_known" for g in aligned_gt),
        "aligned_novel_tracks": len(novel),
        "aligned_novel_categories": len(by_cat),
        "novel_tracks_by_category": dict(sorted(by_cat.items())),
        "cross_physical_same_category_pairs": pairs,
        "cross_video_same_category_pairs": cross_video,
        "opportunity_gate": {"cross_physical_target": 100, "cross_video_target": 30, "passed": pairs >= 100 and cross_video >= 30},
        "gt_fragmentation_before_one_to_one": sum(max(0, n - 1) for n in gt_to_pred.values()),
        "unmatched_proposal_tracks": len(pred_by_key) - len(mapping),
        "persistent_false_proposal_rows": sum(1 for r in rows if (r["video_id"], r["track_id"]) not in mapping and r["prior_hits"] >= 2),
        "source_categories_used_for_model": False,
        "gt_labels_used_for_alignment_only": True,
        "future_frames_used": False,
        "q1_label_used": False,
    }
    out_a = ROOT / args.out_audit; out_a.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_a.with_suffix(out_a.suffix + ".tmp"); tmp.write_text(json.dumps(audit, indent=2, sort_keys=True)); os.replace(tmp, out_a)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
