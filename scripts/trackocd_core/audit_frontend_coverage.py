#!/usr/bin/env python3
"""Evaluator-only Base/Novel coverage of the fixed PANDAS ByteTrack stream.

Same frozen full-Val input, fixed legacy temporal-IoU >= .5 and one-to-one
per-video Hungarian. No model, inference, thresholds sweep or memory update.
Only canonical annotated frames enter geometric matching; ByteTrack itself
already processed all intervening video frames. Labels only aggregate scores.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.trackocd_core.audit_assets import DATASET, file_record, read_annotation, validate_roles
from scripts.trackocd_core.audit_frontend import EVAL_ROOT, TRACK_ROOT
from src.trackocd_v2.io import atomic_json

THRESHOLD = .5


def pairwise_frame_iou(box, candidates):
    candidates = np.asarray(candidates, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    intersection = np.maximum(0, np.minimum(box[2:], candidates[:, 2:]) - np.maximum(box[:2], candidates[:, :2])).prod(axis=1)
    area = np.maximum(0, box[2:] - box[:2]).prod()
    pred_area = np.maximum(0, candidates[:, 2:] - candidates[:, :2]).prod(axis=1)
    union = area + pred_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_video(gt_tracks: list[dict], frames: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Geometry only; source IDs are local and never interpreted as categories."""
    all_ids = np.concatenate([ids for ids, _ in frames.values()]) if frames else np.array([], dtype=np.int64)
    ids, counts = np.unique(all_ids, return_counts=True)
    prepared = {}
    for frame, (local_ids, boxes) in frames.items():
        if len(np.unique(local_ids)) != len(local_ids):
            raise ValueError("Duplicate physical track in a canonical frame")
        prepared[frame] = (np.searchsorted(ids, local_ids), boxes)
    scores = np.zeros((len(gt_tracks), len(ids)), dtype=np.float64)
    for row, gt in enumerate(gt_tracks):
        common = np.zeros(len(ids), dtype=np.int64)
        sums = np.zeros(len(ids), dtype=np.float64)
        for image_id, box in gt["boxes"].items():
            if image_id not in prepared:
                continue
            indices, boxes = prepared[image_id]
            common[indices] += 1
            sums[indices] += pairwise_frame_iou(box, boxes)
        denominator = len(gt["boxes"]) + counts - common
        scores[row] = np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator > 0)
    matched = {}
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        matched = {gt_tracks[int(row)]["key"]: {
            "temporal_iou": float(scores[row, column]),
            "predicted_local_track_id": int(ids[column]),
            "reliable": bool(scores[row, column] >= THRESHOLD),
        } for row, column in zip(rows, columns)}
    return {"matches": matched, "annotated_predicted_tracks": len(ids),
            "annotated_prediction_rows": len(all_ids)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/trackocd_core/audit/frontend_coverage.json")
    args = parser.parse_args()
    started = time.monotonic()
    mem = {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())}
    if mem["MemAvailable"] < mem["MemTotal"] * .25:
        raise RuntimeError("Resource wait: less than 25% RAM headroom")
    roles_path = ROOT / "configs/trackocd_core/roles.json"
    roles = json.loads(roles_path.read_text())
    validate_roles(roles)
    known, novel, distractor = [set(roles[k]) for k in ("known_ids", "novel_ids", "distractor_ids")]
    annotation_path = DATASET / "annotations/validation.json"
    annotation = read_annotation(annotation_path, "val")
    image_to_video = {int(im["id"]): int(im["video_id"]) for im in annotation["images"]}
    grouped = {}
    for ann in annotation["annotations"]:
        category = int(ann["category_id"])
        if category in distractor:
            continue
        if category not in known | novel:
            raise ValueError("Category outside inherited roles")
        video_id, track_id = int(ann["video_id"]), int(ann["track_id"])
        key = f"{video_id}_{track_id}"
        row = grouped.setdefault(key, {"key": key, "video_id": video_id,
                                       "category": category, "role": "known" if category in known else "novel", "boxes": {}})
        if row["category"] != category:
            raise ValueError("Inconsistent GT track category")
        x, y, w, h = ann["bbox"]
        row["boxes"][int(ann["image_id"])] = [x, y, x + w, y + h]
    by_video = defaultdict(list)
    for row in grouped.values():
        by_video[row["video_id"]].append(row)
    paths = sorted(TRACK_ROOT.glob("video_*.npz"))
    if {int(p.stem.removeprefix("video_")) for p in paths} != {int(v["id"]) for v in annotation["videos"]}:
        raise ValueError("Not the full frozen Val video universe")
    results = {}
    annotated_rows = annotated_frames = annotated_tracks = 0
    for path in paths:
        video_id = int(path.stem.removeprefix("video_"))
        with np.load(path, allow_pickle=False) as data:
            image_ids, offsets = data["image_id"], data["frame_offsets"]
            local_ids, boxes = data["track_id"], data["boxes"]
            frames = {}
            for index, raw_id in enumerate(image_ids):
                image_id = int(raw_id)
                if image_id not in image_to_video:
                    if image_id != -1:
                        raise ValueError("Unregistered canonical image ID")
                    continue
                if image_to_video[image_id] != video_id or image_id in frames:
                    raise ValueError("Canonical image/video identity mismatch")
                begin, end = int(offsets[index]), int(offsets[index + 1])
                frames[image_id] = (local_ids[begin:end], boxes[begin:end])
            match = match_video(by_video[video_id], frames)
            results.update(match["matches"])
            annotated_rows += match["annotated_prediction_rows"]
            annotated_tracks += match["annotated_predicted_tracks"]
            annotated_frames += len(frames)
    export_path = EVAL_ROOT / "export_trackeval_complete.json"
    export = json.loads(export_path.read_text())
    if annotated_rows != export["track_rows_exported"] or annotated_frames != export["annotated_frames_exported"]:
        raise ValueError("Canonical NPZ projection differs from frozen TrackEval export")
    by_role = {}
    for role in ("known", "novel"):
        rows = [row for row in grouped.values() if row["role"] == role]
        observed = sum(results.get(row["key"], {}).get("reliable", False) for row in rows)
        by_role[role] = {"reliably_observed": observed, "gt_tracks": len(rows),
                         "coverage": observed / len(rows) if rows else None,
                         "missing_or_unreliable": len(rows) - observed}
    result = {
        "schema_version": "trackocd.core.frozen-frontend-coverage.v1",
        "status": "EVALUATOR_DIAGNOSTIC_COMPLETE",
        "role": "FROZEN_PANDAS_BT_FG0_EXTERNAL_REFERENCE_NOT_CORE_FRONTEND_FREEZE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "videos": len(paths), "annotated_frames": annotated_frames,
        "annotated_prediction_rows": annotated_rows, "annotated_prediction_tracks": annotated_tracks,
        "non_distractor_gt_tracks": len(grouped), "by_role": by_role,
        "fixed_geometric_join": {"threshold": THRESHOLD, "assignment": "one global per-video Hungarian over physical tracks, category-free geometry",
                                 "temporal_iou": "sum(box IoU on common canonical image IDs) / count(union of GT/pred canonical image IDs)",
                                 "same_rule_as_legacy_v2_frontend_observability": True,
                                 "unannotated_frames": "used by original association, excluded from annotation-frame geometric scoring"},
        "annotation": file_record(annotation_path), "roles": file_record(roles_path),
        "frozen_export_evidence": file_record(export_path),
        "projection_counts_agree_with_frozen_export": True,
        "projected_box_bytes_compared_with_export": False,
        "full_npz_checksums_verified": False,
        "labels_used_only_to_aggregate_geometry_matches": True,
        "model_or_memory_received_gt": False,
        "thresholds_tuned": False,
        "matched_only_ocd_score_computed": False,
        "training_started": False, "test_data_accessed": False,
        "external_process_interference": False,
        "resources": {"worker_count": 1, "gpu_used": False, "initial_mem_available_kib": mem["MemAvailable"],
                      "ram_plan": "one video of existing arrays; <2 GiB working plan; preserve 25% system headroom",
                      "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "wall_seconds": time.monotonic() - started},
    }
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "by_role": by_role, "wall_seconds": result["resources"]["wall_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
