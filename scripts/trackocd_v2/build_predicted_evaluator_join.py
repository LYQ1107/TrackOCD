#!/usr/bin/env python3
"""Build the evaluator-only IoU join for the public predicted stream.

Prediction decisions are made on all public predicted tracks before this
script is called.  This script uses only geometry to match the small,
historical IoU>=0.5 diagnostic subset back to GT tracks, then attaches the
private category role for scoring.  The resulting join is never passed to a
method or controller.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evaluation.track_matching import temporal_iou  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402


PRED_PUBLIC = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream.jsonl"
PRED_DIAGNOSTIC = ROOT / "data/tao_ow_ocd_v1/public/pred_track_stream_matched_iou0.5.jsonl"
GT_MANIFEST = OUTPUT_TARGET / "manifests/tao_val_gt_tracks.jsonl"
GT_LABELS = OUTPUT_TARGET / "manifests/private_tao_val_gt_track_labels.jsonl"
JOIN_PATH = OUTPUT_TARGET / "manifests/tao_val_predicted_evaluator_join.jsonl"
AUDIT_PATH = OUTPUT_TARGET / "audit/predicted_evaluator_join.json"
THRESHOLD = 0.5


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _track_boxes(row: dict) -> dict[int, list[float]]:
    frames = [int(value) for value in row["frame_ids"]]
    boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
    if len(frames) != len(boxes) or len(set(frames)) != len(frames):
        raise ValueError(f"invalid frame/box lineage for {row.get('sample_key', row.get('sample_id'))}")
    return dict(zip(frames, boxes))


def _gt_groups() -> tuple[dict[int, list[dict]], dict[str, dict]]:
    labels = {str(row["sample_key"]): row for row in _read_jsonl(GT_LABELS)}
    by_video: defaultdict[int, list[dict]] = defaultdict(list)
    for row in _read_jsonl(GT_MANIFEST):
        key = str(row["sample_key"])
        label = labels.get(key)
        if label is None:
            raise ValueError(f"GT label sidecar missing {key}")
        if bool(label.get("is_distractor", False)) or str(label.get("gt_split")) == "distractor":
            continue
        by_video[int(row["video_id"])].append({
            "sample_key": key,
            "video_id": int(row["video_id"]),
            "physical_track_id": str(row["physical_track_id"]),
            "boxes": _track_boxes(row),
            "gt_category_id": int(label["gt_category_id"]),
            "gt_split": str(label["gt_split"]),
        })
    return dict(by_video), labels


def _pred_groups() -> tuple[dict[int, list[dict]], int]:
    by_video: defaultdict[int, list[dict]] = defaultdict(list)
    with PRED_DIAGNOSTIC.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = {"gt_category_id", "gt_split", "gt_category_name", "gt_match_id"}
            if forbidden & set(row):
                raise ValueError(f"public predicted row contains evaluator fields: {row.get('sample_id')}")
            by_video[int(row["video_id"])].append({
                "sample_key": "pred_" + str(row["sample_id"]),
                "source_sample_id": str(row["sample_id"]),
                "video_id": int(row["video_id"]),
                "physical_track_id": str(row["track_id"]),
                "stream_order": int(row.get("stream_order", 0)),
                "boxes": _track_boxes(row),
            })
    return dict(by_video), sum(len(rows) for rows in by_video.values())


def _build_matches(gt_by_video: dict[int, list[dict]], pred_by_video: dict[int, list[dict]]) -> list[dict]:
    matched: list[dict] = []
    for video_id in sorted(pred_by_video):
        predictions = pred_by_video[video_id]
        gts = gt_by_video.get(video_id, [])
        if not gts:
            continue
        scores = np.zeros((len(predictions), len(gts)), dtype=np.float64)
        for p_index, pred in enumerate(predictions):
            for g_index, gt in enumerate(gts):
                scores[p_index, g_index] = temporal_iou(gt["boxes"], pred["boxes"])
        pred_indices, gt_indices = linear_sum_assignment(-scores)
        for p_index, g_index in zip(pred_indices, gt_indices):
            score = float(scores[p_index, g_index])
            if score < THRESHOLD:
                continue
            pred = predictions[int(p_index)]
            gt = gts[int(g_index)]
            matched.append({
                "sample_key": pred["sample_key"],
                "source_sample_id": pred["source_sample_id"],
                "video_id": pred["video_id"],
                "physical_track_id": pred["physical_track_id"],
                "stream_order": pred["stream_order"],
                "gt_sample_key": gt["sample_key"],
                "gt_category_id": gt["gt_category_id"],
                "gt_split": gt["gt_split"],
                "temporal_iou": score,
                "match_threshold": THRESHOLD,
                "evaluator_only": True,
            })
    matched.sort(key=lambda row: (int(row["stream_order"]), int(row["video_id"]), str(row["sample_key"])))
    return matched


def main() -> int:
    out = ensure_output_layout()
    for path in (PRED_PUBLIC, PRED_DIAGNOSTIC, GT_MANIFEST, GT_LABELS):
        if not path.exists():
            raise FileNotFoundError(path)
    gt_by_video, _ = _gt_groups()
    pred_by_video, diagnostic_count = _pred_groups()
    matches = _build_matches(gt_by_video, pred_by_video)

    public_count = sum(1 for line in PRED_PUBLIC.open(encoding="utf-8") if line.strip())
    gt_count = sum(len(rows) for rows in gt_by_video.values())
    old = sum(row["gt_split"] == "old" for row in matches)
    new = sum(row["gt_split"] == "new" for row in matches)
    matched_gt = {row["gt_sample_key"] for row in matches}
    temporary_text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in matches)
    atomic_write_text(JOIN_PATH, temporary_text)
    audit = {
        "schema_version": "trackocd.v2.predicted_evaluator_join.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "public_prediction_source": str(PRED_PUBLIC.resolve()),
        "public_prediction_source_sha256": sha256_file(PRED_PUBLIC),
        "historical_diagnostic_source": str(PRED_DIAGNOSTIC.resolve()),
        "historical_diagnostic_source_sha256": sha256_file(PRED_DIAGNOSTIC),
        "gt_manifest": str(GT_MANIFEST.resolve()),
        "gt_manifest_sha256": sha256_file(GT_MANIFEST),
        "public_prediction_tracks": public_count,
        "diagnostic_subset_tracks": diagnostic_count,
        "matched_evaluator_tracks": len(matches),
        "gt_non_distractor_tracks": gt_count,
        "gt_track_coverage_within_non_distractor": len(matched_gt) / max(gt_count, 1),
        "matched_old": old,
        "matched_new": new,
        "stream_order_preserved": all(
            int(left["stream_order"]) <= int(right["stream_order"])
            for left, right in zip(matches, matches[1:])
        ),
        "matching": {
            "method": "temporal bbox IoU plus per-video Hungarian",
            "threshold": THRESHOLD,
            "category_free": True,
            "performed_after_public_decisions": True,
            "historical_subset_is_not_model_input": True,
        },
        "join_path": str(JOIN_PATH.resolve()),
        "join_sha256": sha256_file(JOIN_PATH),
        "test_semantic_accessed": False,
    }
    atomic_json(AUDIT_PATH, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
