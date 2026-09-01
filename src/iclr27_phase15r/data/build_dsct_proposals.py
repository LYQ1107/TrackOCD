"""Convert the frozen Phase-6B DSCT subset output into a causal proposal CSV.

The detector output is matched to the public TRAIN annotations only for bank
construction/audit.  No category label is passed to the detector or used by
the online Q1 replay.  Matching is frame-local (IoU >= .5) and never uses a
future frame or a private annotation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
KNOWN_PATH = ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json"


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + max(0.0, aw), ay1 + max(0.0, ah)
    bx2, by2 = bx1 + max(0.0, bw), by1 + max(0.0, bh)
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0 else 0.0


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def convert(annotation_path: Path, track_path: Path, out_path: Path) -> dict:
    ann = json.loads(annotation_path.read_text())
    detections = json.loads(track_path.read_text())
    known = {int(x) for x in json.loads(KNOWN_PATH.read_text())}
    images = {int(x["id"]): x for x in ann["images"]}
    gt_by_image: dict[int, list[dict]] = defaultdict(list)
    for a in ann.get("annotations", []):
        if a.get("iscrowd"):
            continue
        bb = [float(v) for v in a["bbox"]]
        gt_by_image[int(a["image_id"])].append({
            "bbox": bb,
            "category_id": int(a["category_id"]),
            "track_id": int(a.get("track_id", -1)),
        })

    # A DSCT row is retained even when it has no matching public object.  This
    # makes false proposals visible in the domain audit and preserves the exact
    # physical stream produced by the frozen detector.
    rows: list[dict] = []
    image_counts: Counter[int] = Counter()
    role_counts: Counter[str] = Counter()
    matched_categories: Counter[int] = Counter()
    matched_tracks: set[tuple[int, int]] = set()
    for det in detections:
        image_id = int(det["image_id"])
        if image_id not in images:
            continue
        im = images[image_id]
        db = [float(v) for v in det["bbox"]]
        candidates = gt_by_image.get(image_id, [])
        best = max(candidates, key=lambda g: iou_xywh(db, g["bbox"])) if candidates else None
        best_iou = iou_xywh(db, best["bbox"]) if best is not None else 0.0
        if best is not None and best_iou >= 0.5:
            gt_cat = int(best["category_id"])
            gt_track = int(best["track_id"])
            role = "known" if gt_cat in known else "novel"
            matched_categories[gt_cat] += 1
            matched_tracks.add((int(im["video_id"]), gt_track))
        else:
            gt_cat, gt_track, role = -1, -1, "fp"
        # Local IDs are unique within a frame, which is the contract consumed
        # by the DINO extraction and chronological replay code.
        proposal_local_id = image_counts[image_id]
        image_counts[image_id] += 1
        rows.append({
            "video_id": int(im["video_id"]),
            "frame_id": int(im["frame_id"]),
            "source_frame_index": int(im["frame_index"]),
            "image_id": image_id,
            "proposal_local_id": proposal_local_id,
            "track_id": int(det["track_id"]),
            "score": float(det.get("score", 0.0)),
            "bbox_xyxy": json.dumps([
                db[0], db[1], db[0] + db[2], db[1] + db[3]
            ], separators=(",", ":")),
            "det_category_id": int(det.get("category_id", -1)),
            "source_family": "phase6b_dsct_subset",
            "prior_hits": 0,
            "gt_track_id": gt_track,
            "gt_category_id": gt_cat,
            "gt_role": role,
            "gt_iou": float(best_iou),
        })
        role_counts[role] += 1

    rows.sort(key=lambda r: (
        int(r["video_id"]), int(r["frame_id"]),
        int(r["proposal_local_id"]), int(r["track_id"])
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)
    summary = {
        "annotation": str(annotation_path.resolve()),
        "track_output": str(track_path.resolve()),
        "proposal_csv": str(out_path.resolve()),
        "rows": len(rows),
        "videos": len({int(r["video_id"]) for r in rows}),
        "physical_tracks": len({(int(r["video_id"]), int(r["track_id"])) for r in rows}),
        "role_counts": dict(sorted(role_counts.items())),
        "matched_categories": dict(sorted(matched_categories.items())),
        "matched_physical_tracks": len(matched_tracks),
        "iou_threshold": 0.5,
        "gt_labels_used_for_alignment_only": True,
        "q1_label_used": False,
        "future_frames_used": False,
        "physical_id_used_as_feature": False,
    }
    atomic_json(out_path.with_name("proposals_summary.json"), summary)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation", default="data/iclr27_phase15r/sources/validation_train_subset.json")
    ap.add_argument("--track-output", default="outputs/iclr27_phase15r/dsct_subset/teta_results/tao_track.json")
    ap.add_argument("--out", default="outputs/iclr27_phase15r/dsct_subset/proposals.csv")
    args = ap.parse_args()
    summary = convert(ROOT / args.annotation, ROOT / args.track_output, ROOT / args.out)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
