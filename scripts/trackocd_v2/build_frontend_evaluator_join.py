#!/usr/bin/env python3
"""Build an evaluator-only join for a selected frontend physical stream.

The selected frontend has already received causal decisions before this
script is called.  Only then are Val GT tracks used to find one-to-one
temporal-IoU matches for OCD scoring.  Observations are spooled to SQLite so
the full physical stream is never accumulated in Python memory.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout, sha256_file  # noqa: E402
from src.evaluation.track_matching import temporal_iou  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FINAL_FREEZE = OUTPUT_TARGET / "audit/FINAL_FREEZE.json"
THRESHOLD = 0.5
FORBIDDEN_FIELDS = {"category_id", "category_name", "text", "gt_category_id", "gt_split", "gt_match_id"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            yield value


def _track_boxes(row: dict[str, Any], context: str) -> dict[int, list[float]]:
    frames = [int(value) for value in row["frame_ids"]]
    boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
    if not frames or len(frames) != len(boxes) or len(set(frames)) != len(frames):
        raise ValueError(f"invalid frame/box lineage for {context}")
    for box in boxes:
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"invalid box for {context}")
    return dict(zip(frames, boxes))


def _load_gt_tracks(manifest_path: Path, labels_path: Path) -> dict[int, list[dict[str, Any]]]:
    labels = {str(row["sample_key"]): row for row in _jsonl(labels_path)}
    by_video: collections.defaultdict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _jsonl(manifest_path):
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
            "boxes": _track_boxes(row, key),
            "gt_category_id": int(label["gt_category_id"]),
            "gt_split": str(label["gt_split"]),
            "stream_order": int(row.get("stream_order", len(by_video[int(row["video_id"])]))),
        })
    if not by_video:
        raise ValueError(f"GT manifest has no non-distractor tracks: {manifest_path}")
    return dict(by_video)


def _create_spool() -> tuple[sqlite3.Connection, Path]:
    fd, path = tempfile.mkstemp(prefix="trackocd_frontend_join_", suffix=".sqlite", dir="/tmp")
    os.close(fd)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE tracks (video_id INTEGER, physical_track_id TEXT, sample_key TEXT, source_sample_id TEXT, stream_order INTEGER, PRIMARY KEY(video_id, physical_track_id))"
    )
    connection.execute(
        "CREATE TABLE observations (video_id INTEGER, physical_track_id TEXT, image_id INTEGER, x1 REAL, y1 REAL, x2 REAL, y2 REAL, PRIMARY KEY(video_id, physical_track_id, image_id))"
    )
    connection.execute("CREATE INDEX observations_by_video ON observations(video_id, physical_track_id, image_id)")
    connection.execute("CREATE INDEX tracks_by_order ON tracks(video_id, stream_order)")
    return connection, Path(path)


def _spool_physical_stream(path: Path, connection: sqlite3.Connection) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    track_count = observation_count = 0
    videos: set[int] = set()
    previous_order = -1
    seen_samples: set[str] = set()
    for row_number, row in enumerate(_jsonl(path), 1):
        if FORBIDDEN_FIELDS & set(row):
            raise ValueError(f"physical stream contains forbidden fields at row {row_number}")
        for field in ("sample_key", "source_sample_id", "video_id", "physical_track_id", "frame_ids", "boxes_xyxy", "stream_order"):
            if field not in row:
                raise ValueError(f"physical stream row {row_number} misses {field}")
        sample_key = str(row["sample_key"])
        if sample_key in seen_samples:
            raise ValueError(f"duplicate physical sample_key: {sample_key}")
        seen_samples.add(sample_key)
        video_id = int(row["video_id"])
        physical_id = str(row["physical_track_id"])
        stream_order = int(row["stream_order"])
        if stream_order < previous_order:
            raise ValueError(f"physical stream order regressed at row {row_number}")
        previous_order = stream_order
        boxes = _track_boxes(row, sample_key)
        try:
            connection.execute(
                "INSERT INTO tracks VALUES (?, ?, ?, ?, ?)",
                (video_id, physical_id, sample_key, str(row["source_sample_id"]), stream_order),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate physical track key: {video_id}/{physical_id}") from exc
        for image_id, box in boxes.items():
            try:
                connection.execute(
                    "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (video_id, physical_id, image_id, *box),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"duplicate physical observation: {video_id}/{physical_id}/{image_id}") from exc
        track_count += 1
        observation_count += len(boxes)
        videos.add(video_id)
        if row_number % 5000 == 0:
            connection.commit()
    connection.commit()
    if not track_count:
        raise ValueError(f"physical stream is empty: {path}")
    return track_count, observation_count, len(videos)


def _match(
    connection: sqlite3.Connection,
    gt_by_video: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    video_ids = sorted(set(gt_by_video) | {int(row[0]) for row in connection.execute("SELECT DISTINCT video_id FROM tracks")})
    for video_id in video_ids:
        gt_rows = gt_by_video.get(video_id, [])
        predictions: list[dict[str, Any]] = []
        for physical_id, sample_key, source_sample_id, stream_order in connection.execute(
            "SELECT physical_track_id, sample_key, source_sample_id, stream_order FROM tracks WHERE video_id = ? ORDER BY stream_order, physical_track_id",
            (video_id,),
        ):
            observations = {
                int(image_id): [float(x1), float(y1), float(x2), float(y2)]
                for image_id, x1, y1, x2, y2 in connection.execute(
                    "SELECT image_id, x1, y1, x2, y2 FROM observations WHERE video_id = ? AND physical_track_id = ? ORDER BY image_id",
                    (video_id, str(physical_id)),
                )
            }
            predictions.append({
                "physical_track_id": str(physical_id),
                "sample_key": str(sample_key),
                "source_sample_id": str(source_sample_id),
                "stream_order": int(stream_order),
                "boxes": observations,
            })
        if not gt_rows or not predictions:
            continue
        scores = np.zeros((len(gt_rows), len(predictions)), dtype=np.float64)
        for gt_index, gt in enumerate(gt_rows):
            for pred_index, pred in enumerate(predictions):
                scores[gt_index, pred_index] = temporal_iou(gt["boxes"], pred["boxes"])
        gt_indices, pred_indices = linear_sum_assignment(-scores)
        for gt_index, pred_index in zip(gt_indices, pred_indices):
            score = float(scores[int(gt_index), int(pred_index)])
            if score < THRESHOLD:
                continue
            gt = gt_rows[int(gt_index)]
            pred = predictions[int(pred_index)]
            matches.append({
                "sample_key": pred["sample_key"],
                "source_sample_id": pred["source_sample_id"],
                "video_id": video_id,
                "physical_track_id": pred["physical_track_id"],
                "predicted_physical_track_id": pred["physical_track_id"],
                "stream_order": pred["stream_order"],
                "gt_sample_key": gt["sample_key"],
                "gt_physical_track_id": gt["physical_track_id"],
                "evaluator_track_key": gt["sample_key"],
                "gt_category_id": gt["gt_category_id"],
                "gt_split": gt["gt_split"],
                "temporal_iou": score,
                "match_threshold": THRESHOLD,
                "evaluator_only": True,
            })
    matches.sort(key=lambda row: (int(row["stream_order"]), int(row["video_id"]), str(row["sample_key"])))
    return matches


def run(
    *,
    physical_stream: Path,
    frontend: str,
    join_path: Path,
    audit_path: Path,
    split: str = "val",
    gt_manifest: Path | None = None,
    gt_labels: Path | None = None,
) -> dict[str, Any]:
    if not frontend:
        raise ValueError("frontend slug is required")
    if split not in {"val", "test"}:
        raise ValueError(f"unsupported evaluator split: {split}")
    if split == "test":
        # This is the only point at which the final Test label sidecar is
        # allowed to enter the evaluator-only join.  All causal decisions
        # must already be sealed by the caller.
        assert_test_semantic_access_allowed(FINAL_FREEZE, "build Test evaluator-only frontend join")
    manifest_path = (gt_manifest or (MANIFEST_ROOT / f"tao_{split}_gt_tracks.jsonl")).resolve()
    labels_path = (gt_labels or (MANIFEST_ROOT / f"private_tao_{split}_gt_track_labels.jsonl")).resolve()
    if not manifest_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"missing {split} GT manifest/labels: {manifest_path}, {labels_path}")
    gt_by_video = _load_gt_tracks(manifest_path, labels_path)
    connection, temporary_path = _create_spool()
    try:
        track_count, observation_count, video_count = _spool_physical_stream(physical_stream, connection)
        matches = _match(connection, gt_by_video)
    finally:
        connection.close()
        if temporary_path.exists():
            temporary_path.unlink()

    atomic_write_text(join_path, "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in matches))
    matched_old = sum(row["gt_split"] == "old" for row in matches)
    matched_new = sum(row["gt_split"] == "new" for row in matches)
    gt_count = sum(len(rows) for rows in gt_by_video.values())
    audit = {
        "schema_version": "trackocd.v2.frontend_evaluator_join.v1",
        "status": "COMPLETE",
        "generated_utc": _now(),
        "frontend": frontend,
        "split": split,
        "gt_manifest": str(manifest_path),
        "gt_labels": str(labels_path),
        "source_physical_stream": str(physical_stream.resolve()),
        "source_physical_stream_sha256": sha256_file(physical_stream),
        "physical_stream_tracks": track_count,
        "physical_stream_observations": observation_count,
        "physical_stream_videos": video_count,
        "gt_non_distractor_tracks": gt_count,
        "matched_evaluator_tracks": len(matches),
        "matched_old": matched_old,
        "matched_new": matched_new,
        "gt_track_coverage_within_non_distractor": len({row["gt_sample_key"] for row in matches}) / max(gt_count, 1),
        "matching": {
            "method": "temporal bbox IoU plus per-video Hungarian",
            "threshold": THRESHOLD,
            "category_free": True,
            "performed_after_causal_decisions": True,
            "historical_subset_is_not_model_input": True,
        },
        "decisions_sealed_before_join": True,
        "gt_join_used_for_model_or_native_stream": False,
        "join_path": str(join_path.resolve()),
        "join_sha256": sha256_file(join_path),
        "test_semantic_accessed": split == "test",
    }
    atomic_json(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-stream", type=Path, required=True)
    parser.add_argument("--frontend", required=True)
    parser.add_argument("--join-path", type=Path, required=True)
    parser.add_argument("--audit-path", type=Path, required=True)
    parser.add_argument("--decisions-sealed", action="store_true", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--gt-manifest", type=Path, default=None)
    parser.add_argument("--gt-labels", type=Path, default=None)
    args = parser.parse_args()
    ensure_output_layout()
    try:
        result = run(
            physical_stream=args.physical_stream.resolve(),
            frontend=args.frontend,
            join_path=args.join_path.resolve(),
            audit_path=args.audit_path.resolve(),
            split=args.split,
            gt_manifest=args.gt_manifest,
            gt_labels=args.gt_labels,
        )
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.frontend_evaluator_join.v1",
            "status": "FAILED_FRONTEND_EVALUATOR_JOIN",
            "generated_utc": _now(),
            "frontend": args.frontend,
            "split": args.split,
            "source_physical_stream": str(args.physical_stream.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "decisions_sealed_before_join": bool(args.decisions_sealed),
            "test_semantic_accessed": False,
        }
        atomic_json(args.audit_path.resolve(), failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
