"""Frozen physical frontend audit (Q1 vs Q2-alpha0.1) and pred->GT alignment."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    Q2_DEV,
    TAO_VAL_ANN,
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
    temporal_iou,
)


def gt_track_boxes(stream_rows: list[dict]) -> dict[int, dict[int, dict[int, list[float]]]]:
    """GT track boxes keyed by TAO image_id (via file_name -> image_id map)."""
    val = json.loads(TAO_VAL_ANN.read_text())
    path_to_img = {im["file_name"]: int(im["id"]) for im in val["images"]}
    out: dict[int, dict[int, dict[int, list[float]]]] = {}
    for r in stream_rows:
        vid = int(r["video_id"])
        boxes = out.setdefault(vid, {})
        for path, b in zip(r["image_paths"], r["boxes_xyxy"]):
            img_id = path_to_img.get(path)
            if img_id is None:
                continue
            boxes.setdefault(int(r["track_id"]), {})[img_id] = [float(v) for v in b]
    return out


def align_pred_to_gt(
    tracks: dict[tuple[int, int], list[dict]],
    gt_boxes: dict[int, dict[int, dict[int, list[float]]]],
) -> dict[tuple[int, int], str]:
    """Greedy one-to-one pred-physical-track -> GT sample_id mapping by temporal IoU."""
    cands = []
    for (vid, tid), rows in tracks.items():
        if vid not in gt_boxes:
            continue
        pred_boxes: dict[int, list[float]] = {}
        for r in rows:
            b = json.loads(r["bbox_xyxy"])
            pred_boxes[int(r["image_id"])] = [float(v) for v in b]
        best_sid, best_iou = None, 0.0
        for gtid, boxes in gt_boxes[vid].items():
            v = temporal_iou(boxes, pred_boxes)
            if v > best_iou:
                best_iou, best_sid = v, f"{vid}_{gtid}"
        if best_sid is not None and best_iou > 0:
            cands.append((best_iou, (vid, tid), best_sid))
    cands.sort(reverse=True)
    used_pred, used_gt = set(), set()
    mapping: dict[tuple[int, int], str] = {}
    for iou, pk, sid in cands:
        if pk in used_pred or sid in used_gt:
            continue
        used_pred.add(pk)
        used_gt.add(sid)
        mapping[pk] = sid
    return mapping


def audit(path: Path) -> dict:
    rows = load_proposals(path)
    n_frames = len(set((r["video_id"], r["frame_id"]) for r in rows))
    roles = Counter(r["gt_role"] for r in rows)
    pfp = sum(1 for r in rows if r["gt_role"] == "fp" and r["prior_hits"] >= 2)
    tracks = group_tracks(rows)
    return {
        "rows": len(rows),
        "frames": n_frames,
        "rows_per_frame": round(len(rows) / max(n_frames, 1), 2),
        "roles": dict(roles),
        "physical_tracks": len(tracks),
        "persistent_fp_per_frame": round(pfp / max(n_frames, 1), 2),
        "tracks": tracks,
        "rows_list": rows,
    }


def main() -> None:
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gt_boxes = gt_track_boxes(stream)
    gt_by_role = Counter(r["protocol_role"] for r in labels.values())
    print("GT dev tracks:", len(labels), dict(gt_by_role))

    report = {}
    for name, path in (("Q1", Q1_DEV), ("Q2-alpha0.1", Q2_DEV)):
        a = audit(path)
        mapping = align_pred_to_gt(a["tracks"], gt_boxes)
        covered = {sid: pk for pk, sid in mapping.items()}
        by_role = defaultdict(int)
        for sid in labels:
            if sid in covered:
                by_role[labels[sid]["protocol_role"]] += 1
        coverage = {
            role: round(by_role.get(role, 0) / max(gt_by_role.get(role, 0), 1), 4)
            for role in ("supported_known", "novel")
        }
        report[name] = {
            "rows": a["rows"],
            "frames": a["frames"],
            "rows_per_frame": a["rows_per_frame"],
            "roles": a["roles"],
            "physical_tracks": a["physical_tracks"],
            "persistent_fp_per_frame": a["persistent_fp_per_frame"],
            "covered_gt_tracks": {k: v for k, v in by_role.items()},
            "coverage": coverage,
        }
        print(f"\n== {name} ==")
        for k, v in report[name].items():
            print(f"  {k}: {v}")
    out = Path("outputs/iclr27_phase4s/frontend_audit")
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2))
    print("\nwrote", out / "audit.json")


if __name__ == "__main__":
    main()
