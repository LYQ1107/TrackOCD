#!/usr/bin/env python3
"""Bounded, category-free integrity audit for the recovered NAS Val streams.

This is deliberately separate from the formal v2 admission path.  It reads
only the five explicitly staged Val JSONL files, one row at a time, and never
opens a TAO annotation, vocabulary, image, checkpoint, or Test path.  The
output contains counts, hashes, and structural errors only; no row IDs,
category values, boxes, or semantic names are emitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import resource
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_recovery_20261022/val"
DEFAULT_OUTPUT = ROOT / "outputs/trackocd_cvpr/cvpr2/nas_val_stream_audit_20261022.json"

COMMON_FIELDS = {
    "sample_key",
    "physical_track_id",
    "video_id",
    "frame_ids",
    "boxes_xyxy",
    "image_paths",
    "quality",
    "source_split",
}
PREDICTED_FIELDS = COMMON_FIELDS | {
    "areas",
    "scores",
    "source_sample_id",
    "stream_order",
}
FORBIDDEN_FIELD_PARTS = ("category", "semantic", "text", "novel", "label", "gt")


def _rss_bytes() -> int:
    # Linux reports ru_maxrss in KiB; keep the conversion local and explicit.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _peak_rss_bytes() -> int:
    self_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    child_peak = int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) * 1024
    return max(self_peak, child_peak)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_box(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(_finite_number(v) for v in value)
        and float(value[2]) > float(value[0])
        and float(value[3]) > float(value[1])
    )


def _external_sort_duplicate_count(keys_path: Path, temp_dir: Path) -> tuple[int, int]:
    """Return (unique, duplicate) exact key counts with bounded sort memory."""

    sorted_path = temp_dir / "sorted_track_keys.txt"
    subprocess.run(
        ["sort", "--stable", "-S", "32M", "-T", str(temp_dir), "-o", str(sorted_path), str(keys_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    unique = 0
    duplicate = 0
    previous: bytes | None = None
    with sorted_path.open("rb") as handle:
        for raw in handle:
            if raw == previous:
                duplicate += 1
            else:
                unique += 1
                previous = raw
    return unique, duplicate


def _audit_file(path: Path, kind: str) -> tuple[dict[str, Any], set[tuple[int, str]], set[int]]:
    expected = COMMON_FIELDS if kind == "gt_order" else PREDICTED_FIELDS
    digest = hashlib.sha256()
    errors: list[str] = []
    track_keys: set[tuple[int, str]] = set()
    video_ids: set[int] = set()
    source_splits: set[str] = set()
    lines = 0
    blank_lines = 0
    observations = 0
    invalid_boxes = 0
    nonfinite_quality = 0
    nonfinite_scores = 0
    scores_outside_unit = 0
    negative_areas = 0
    invalid_box_reasons = {
        "not_four_values": 0,
        "nonfinite_or_nonnumeric": 0,
        "x2_le_x1": 0,
        "y2_le_y1": 0,
    }
    duplicate_track_rows = 0
    duplicate_frames = 0
    field_mismatch_rows = 0
    max_line_bytes = 0
    previous_stream_order: int | None = None
    stream_order_decreases = 0
    track_count = 0

    # The predicted Val stream has hundreds of thousands of physical tracks.
    # Keep its exact duplicate check in a temporary text index and use GNU sort
    # with a fixed 32 MiB memory cap, rather than retaining all keys in Python.
    dedup_key_handle = None
    dedup_key_path: Path | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if kind == "predicted":
        temp_dir = tempfile.TemporaryDirectory(prefix="trackocd_val_audit_")
        dedup_key_path = Path(temp_dir.name) / "track_keys.txt"
        dedup_key_handle = dedup_key_path.open("wb")

    try:
        with path.open("rb") as handle:
            for raw in handle:
                digest.update(raw)
                max_line_bytes = max(max_line_bytes, len(raw))
                if not raw.strip():
                    blank_lines += 1
                    continue
                lines += 1
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    if len(errors) < 10:
                        errors.append(f"invalid JSON at line {lines}")
                    continue
                if not isinstance(row, dict):
                    if len(errors) < 10:
                        errors.append(f"non-object row at line {lines}")
                    continue

                fields = set(row)
                if fields != expected:
                    field_mismatch_rows += 1
                    if len(errors) < 10:
                        errors.append(f"field schema mismatch at line {lines}")
                for field in fields:
                    if any(part in field.lower() for part in FORBIDDEN_FIELD_PARTS):
                        if len(errors) < 10:
                            errors.append(f"forbidden field name at line {lines}")
                        break

                try:
                    video_id = int(row["video_id"])
                    track_id = str(row["physical_track_id"])
                    frame_ids = row["frame_ids"]
                    boxes = row["boxes_xyxy"]
                    image_paths = row["image_paths"]
                    quality = row["quality"]
                    source_split = str(row["source_split"])
                    source_splits.add(source_split)
                    video_ids.add(video_id)
                    key = (video_id, track_id)
                    if dedup_key_handle is None:
                        if key in track_keys:
                            duplicate_track_rows += 1
                        track_keys.add(key)
                        track_count = len(track_keys)
                    else:
                        encoded_track_id = json.dumps(track_id, ensure_ascii=True, separators=(",", ":"))
                        dedup_key_handle.write(f"{video_id}\t{encoded_track_id}\n".encode("utf-8"))
                    if not isinstance(frame_ids, list) or not isinstance(boxes, list):
                        if len(errors) < 10:
                            errors.append(f"frame/box arrays are not lists at line {lines}")
                        continue
                    if not isinstance(image_paths, list) or not isinstance(quality, list):
                        if len(errors) < 10:
                            errors.append(f"image/quality arrays are not lists at line {lines}")
                        continue
                    lengths = {len(frame_ids), len(boxes), len(image_paths), len(quality)}
                    if kind == "predicted":
                        lengths.update({len(row["scores"]), len(row["areas"])})
                    if len(lengths) != 1:
                        if len(errors) < 10:
                            errors.append(f"aligned array lengths mismatch at line {lines}")
                        continue
                    if len(set(frame_ids)) != len(frame_ids):
                        duplicate_frames += 1
                    if kind == "predicted":
                        order = int(row["stream_order"])
                        if previous_stream_order is not None and order < previous_stream_order:
                            stream_order_decreases += 1
                        previous_stream_order = order
                    for box, q in zip(boxes, quality):
                        observations += 1
                        if not _finite_box(box):
                            invalid_boxes += 1
                            if not isinstance(box, list) or len(box) != 4:
                                invalid_box_reasons["not_four_values"] += 1
                            elif not all(_finite_number(v) for v in box):
                                invalid_box_reasons["nonfinite_or_nonnumeric"] += 1
                            else:
                                if float(box[2]) <= float(box[0]):
                                    invalid_box_reasons["x2_le_x1"] += 1
                                if float(box[3]) <= float(box[1]):
                                    invalid_box_reasons["y2_le_y1"] += 1
                        if not _finite_number(q):
                            nonfinite_quality += 1
                    if kind == "predicted":
                        for score, area in zip(row["scores"], row["areas"]):
                            if not _finite_number(score):
                                nonfinite_scores += 1
                            elif not 0.0 <= float(score) <= 1.0:
                                scores_outside_unit += 1
                            if not _finite_number(area) or float(area) < 0.0:
                                negative_areas += 1
                except (KeyError, TypeError, ValueError):
                    if len(errors) < 10:
                        errors.append(f"malformed structural row at line {lines}")
        if dedup_key_handle is not None:
            dedup_key_handle.close()
            dedup_key_handle = None
            try:
                track_count, duplicate_track_rows = _external_sort_duplicate_count(dedup_key_path, Path(temp_dir.name))
            except (OSError, subprocess.CalledProcessError) as exc:
                errors.append(f"external duplicate sort failed: {type(exc).__name__}")
    finally:
        if dedup_key_handle is not None:
            dedup_key_handle.close()
        if temp_dir is not None:
            temp_dir.cleanup()

    result = {
        "status": "PASS" if not errors and not field_mismatch_rows and not duplicate_track_rows and not duplicate_frames and not invalid_boxes and not nonfinite_quality and not nonfinite_scores and not scores_outside_unit and not negative_areas else "FAIL",
        "path": str(path.resolve()),
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "jsonl_rows": lines,
        "blank_lines": blank_lines,
        "videos": len(video_ids),
        "tracks": track_count,
        "observations": observations,
        "source_split_values": sorted(source_splits),
        "field_mismatch_rows": field_mismatch_rows,
        "duplicate_track_rows": duplicate_track_rows,
        "duplicate_frame_rows": duplicate_frames,
        "invalid_boxes": invalid_boxes,
        "invalid_box_reasons": invalid_box_reasons,
        "nonfinite_quality": nonfinite_quality,
        "nonfinite_scores": nonfinite_scores,
        "scores_outside_0_1": scores_outside_unit,
        "negative_or_nonfinite_areas": negative_areas,
        "stream_order_decreases": stream_order_decreases,
        "max_line_bytes": max_line_bytes,
        "dedup_backend": "temporary_external_sort_32MiB" if kind == "predicted" else "in_memory_set",
        "errors": errors,
    }
    return result, track_keys, video_ids


def audit(root: Path, output: Path) -> dict[str, Any]:
    if "test" in str(root).lower():
        raise ValueError("refusing a root containing 'test'")
    files = {
        "predicted": (root / "tao_val_predicted_tracks.jsonl", "predicted"),
        "base": (root / "tao_val_gt_tracks.jsonl", "gt_order"),
        "seed1027": (root / "tao_val_gt_tracks_seed1027.jsonl", "gt_order"),
        "seed1028": (root / "tao_val_gt_tracks_seed1028.jsonl", "gt_order"),
        "seed1029": (root / "tao_val_gt_tracks_seed1029.jsonl", "gt_order"),
    }
    results: dict[str, Any] = {}
    gt_track_sets: dict[str, set[tuple[int, str]]] = {}
    gt_video_sets: dict[str, set[int]] = {}
    for name, (path, kind) in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result, track_keys, video_ids = _audit_file(path, kind)
        results[name] = result
        if kind == "gt_order":
            gt_track_sets[name] = track_keys
            gt_video_sets[name] = video_ids

    base_tracks = gt_track_sets["base"]
    base_videos = gt_video_sets["base"]
    for name in sorted(gt_track_sets):
        results[name]["same_track_key_set_as_base"] = gt_track_sets[name] == base_tracks
        results[name]["same_video_set_as_base"] = gt_video_sets[name] == base_videos
    gt_hashes = [results[name]["sha256"] for name in ("base", "seed1027", "seed1028", "seed1029")]
    aggregate_errors = []
    if len(set(gt_hashes)) != 4:
        aggregate_errors.append("the four causal-order files are not byte-distinct")
    if any(not results[name]["same_track_key_set_as_base"] for name in gt_track_sets):
        aggregate_errors.append("causal-order files do not share the same physical track key set")
    if any(not results[name]["same_video_set_as_base"] for name in gt_video_sets):
        aggregate_errors.append("causal-order files do not share the same video set")
    if any(results[name]["status"] != "PASS" for name in results):
        aggregate_errors.append("one or more stream audits failed")
    peak_rss_bytes = _peak_rss_bytes()
    memory_plan_bytes = 256 * 1024 * 1024
    if peak_rss_bytes > memory_plan_bytes:
        aggregate_errors.append("peak RSS exceeded the 256 MiB audit plan")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "trackocd.cvpr.nas-val-stream-audit.v1",
        "status": "PASS_VAL_ONLY_CATEGORY_FREE_STREAM_INTEGRITY" if not aggregate_errors else "FAIL_VAL_ONLY_STREAM_INTEGRITY",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "files": results,
        "causal_order": {
            "file_count": 4,
            "byte_distinct": len(set(gt_hashes)) == 4,
            "same_track_key_set": not any(not results[name]["same_track_key_set_as_base"] for name in gt_track_sets),
            "same_video_set": not any(not results[name]["same_video_set_as_base"] for name in gt_video_sets),
        },
        "aggregate_errors": aggregate_errors,
        "boundaries": {
            "test_paths_read": False,
            "test_content_opened": False,
            "category_values_decoded": False,
            "semantic_names_decoded": False,
            "images_opened": False,
            "weights_opened": False,
            "training_started": False,
            "formal_metrics_created": False,
            "external_processes_touched": False,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "memory_plan_bytes": memory_plan_bytes,
        "memory_plan_pass": peak_rss_bytes <= memory_plan_bytes,
        "interpretation": "Val-only structural and hash audit of staged streams; not formal v2 admission, frontend qualification, OCD/MOT evaluation, or model selection.",
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
