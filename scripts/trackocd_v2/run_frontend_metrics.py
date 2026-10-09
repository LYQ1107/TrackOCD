#!/usr/bin/env python3
"""Run the shared category-free physical metrics for one frontend stream.

The native stream is converted to the pinned TrackEval TAO-OW format with a
single foreground class.  Native category fields are audited but never used
by this conversion.  A separate evaluator-only pass matches complete
frontend tracks to GT tracks by temporal box IoU and reports the two
observability diagnostics required by the v2 protocol.

This script does not run a model and does not produce semantic decisions.  It
therefore consumes Val annotations only after the native physical stream has
been sealed.  TAO Test is intentionally not opened.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import FRONTENDS, OVTR_REFERENCE_ROLE, route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402
from src.trackocd_v2.protocol import assert_test_semantic_access_allowed  # noqa: E402


MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
GT_MANIFEST = MANIFEST_ROOT / "tao_val_gt_tracks.jsonl"
GT_LABELS = MANIFEST_ROOT / "private_tao_val_gt_track_labels.jsonl"
TRACK_EVAL_GT = ROOT / "third_party/TrackEval/data/gt/tao/tao_validation/validation.json"
TETA_ANNOTATION = ROOT / "data/raw/tao/annotations/validation.json"
TEST_MANIFEST = MANIFEST_ROOT / "tao_test_gt_tracks.jsonl"
TEST_LABELS = MANIFEST_ROOT / "private_tao_test_gt_track_labels.jsonl"
TEST_TRACK_EVAL_GT = MANIFEST_ROOT / "tao_test_trackeval/tao_test.json"
TEST_TETA_ANNOTATION = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/masa/data/tao/annotations/tao_test_lvis_v1_classes.json")
TETA_PYTHON = Path(os.environ.get("TRACKOCD_TETA_PYTHON", "/home/lwr/anaconda3/envs/ovtrack/bin/python"))
KNOWN_IDS = ROOT / "data/tao_ow_ocd_v1/splits/known_ids.json"
NOVEL_IDS = ROOT / "data/tao_ow_ocd_v1/splits/unknown_ids_val.json"
STAGE_NAMES = {
    "simowt": "SimOWT/Q0",
    "ovtr": "OVTR-native",
    "covtrack_native": "COVTrack-native",
    "covtrack_nosem": "COVTrack-NoSemantic",
}
OVTR_TETA_FAILURE_STATUS = "OVTR_NATIVE_TETA_REFERENCE_EVAL_FAILED"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _teta_runtime_snapshot() -> dict[str, Any]:
    """Record enough evaluator provenance to diagnose a reference-only failure."""

    script = (ROOT / "scripts/trackocd_v2/run_teta_reference.py").resolve()
    result: dict[str, Any] = {
        "python": str(TETA_PYTHON),
        "python_exists": TETA_PYTHON.is_file(),
        "reference_script": str(script),
        "reference_script_sha256": sha256_file(script) if script.is_file() else None,
    }
    if not TETA_PYTHON.is_file():
        return result
    # Keep the probe self-contained so a broken TETA import still yields the
    # Python and installed-distribution versions in the parent audit.
    probe = (
        "import importlib.metadata as m, json, sys; "
        "names=('teta','numpy','torch'); "
        "versions={name: (m.version(name) if any(d.metadata.get('Name','').lower()==name for d in m.distributions()) else None) for name in names}; "
        "print(json.dumps({'python_version':sys.version, 'versions':versions}, sort_keys=True))"
    )
    try:
        process = subprocess.run(
            [str(TETA_PYTHON), "-c", probe],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        result["probe_returncode"] = int(process.returncode)
        if process.stdout.strip():
            result.update(json.loads(process.stdout.strip().splitlines()[-1]))
        if process.stderr.strip():
            result["probe_stderr_tail"] = process.stderr.strip()[-2000:]
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        result["probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _record_ovtr_teta_failure(
    *,
    exc: Exception,
    run_root: Path,
    native_path: Path,
    split: str,
) -> dict[str, Any]:
    """Turn a native TETA error into a non-blocking OVTR reference record."""

    teta_root = run_root / "teta"
    output_json = teta_root / "teta_reference.json"
    raw_failure: dict[str, Any] | None = None
    if output_json.is_file():
        try:
            candidate = json.loads(output_json.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                raw_failure = candidate
        except (OSError, json.JSONDecodeError):
            raw_failure = None
    record = {
        "schema_version": "trackocd.v2.ovtr_teta_reference_failure.v1",
        "status": OVTR_TETA_FAILURE_STATUS,
        "frontend": "OVTR-native",
        "frontend_role": OVTR_REFERENCE_ROLE,
        "reference_only": True,
        "error": f"{type(exc).__name__}: {exc}",
        "runtime": _teta_runtime_snapshot(),
        "native_input": str(native_path.resolve()),
        "native_input_sha256": sha256_file(native_path),
        "teta_output": str(output_json.resolve()),
        "teta_output_sha256": sha256_file(output_json) if output_json.is_file() else None,
        "raw_teta_failure": raw_failure,
        "physical_metrics_preserved": True,
        "physical_metrics_run_root": str(run_root.resolve()),
        "split": split,
        "test_semantic_accessed": split == "test",
    }
    failure_audit = teta_root / "ovtr_teta_reference_failure.json"
    atomic_json(failure_audit, record)
    record["failure_audit"] = str(failure_audit.resolve())
    record["failure_audit_sha256"] = sha256_file(failure_audit)
    return record


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            yield value


def _box_iou(left: list[float], right: list[float]) -> float:
    if len(left) != 4 or len(right) != 4:
        raise ValueError("boxes must have four coordinates")
    if not all(math.isfinite(value) for value in left + right):
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union > 0.0 else 0.0


def _track_boxes(row: dict[str, Any]) -> dict[int, list[float]]:
    frames = [int(value) for value in row["frame_ids"]]
    boxes = [[float(value) for value in box] for box in row["boxes_xyxy"]]
    if len(frames) != len(boxes) or len(set(frames)) != len(frames):
        raise ValueError(f"invalid frame/box lineage for {row.get('sample_key')}")
    return dict(zip(frames, boxes))


def _load_gt_tracks(
    manifest_path: Path = GT_MANIFEST,
    labels_path: Path = GT_LABELS,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    labels = {str(row["sample_key"]): row for row in _jsonl(labels_path)}
    by_video: collections.defaultdict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    seen: set[str] = set()
    stream_order = 0
    for row in _jsonl(manifest_path):
        key = str(row["sample_key"])
        if key in seen:
            raise ValueError(f"duplicate GT sample key: {key}")
        seen.add(key)
        label = labels.get(key)
        if label is None:
            raise ValueError(f"GT label sidecar missing {key}")
        split = str(label.get("gt_split", ""))
        if split == "distractor" or bool(label.get("is_distractor", False)):
            continue
        record = {
            "sample_key": key,
            "video_id": int(row["video_id"]),
            "physical_track_id": str(row["physical_track_id"]),
            "boxes": _track_boxes(row),
            "gt_category_id": int(label["gt_category_id"]),
            "gt_split": split,
            # The manifest order is the registered causal stream order.  The
            # GT labels are attached only in this evaluator-side structure.
            "stream_order": stream_order,
        }
        stream_order += 1
        by_video[record["video_id"]].append(record)
    if not seen:
        raise ValueError("GT manifest is empty")
    return dict(by_video), labels


def _atomic_json_array(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, int, int, collections.Counter[int]]:
    """Write a JSON array atomically while returning stream counts.

    The array is intentionally kept in the format expected by TrackEval, but
    rows are never accumulated in Python memory.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    videos: set[int] = set()
    track_ids: set[tuple[int, str]] = set()
    categories: collections.Counter[int] = collections.Counter()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[")
            first = True
            for row in rows:
                if not first:
                    handle.write(",")
                first = False
                serialized = dict(row)
                native_category = serialized.pop("_native_category_id", None)
                handle.write(json.dumps(serialized, sort_keys=True, separators=(",", ":"), allow_nan=False))
                count += 1
                videos.add(int(row["video_id"]))
                track_ids.add((int(row["video_id"]), str(row["physical_track_id"])))
                if native_category is not None:
                    categories[int(native_category)] += 1
            handle.write("]")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return count, len(videos), len(track_ids), categories


def _iter_tracker_rows(native_path: Path, spool: sqlite3.Connection) -> Iterator[dict[str, Any]]:
    """Validate native rows, spool them, and yield category-free TAO rows."""

    track_ids: dict[tuple[int, str], int] = {}
    next_track_id = 1
    for row_number, row in enumerate(_jsonl(native_path), 1):
        required = {"video_id", "image_id", "physical_track_id", "bbox_xyxy", "score"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"native row {row_number} missing fields: {sorted(missing)}")
        video_id = int(row["video_id"])
        image_id = int(row["image_id"])
        physical_id = str(row["physical_track_id"])
        box = [float(value) for value in row["bbox_xyxy"]]
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError(f"native row {row_number} has non-finite/malformed bbox")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"native row {row_number} has non-finite score")
        key = (video_id, physical_id)
        if key not in track_ids:
            track_ids[key] = next_track_id
            next_track_id += 1
        try:
            spool.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (video_id, image_id, physical_id, box[0], box[1], box[2], box[3], score, row.get("category_id"), row_number),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate (video,physical_track,image) in native row {row_number}") from exc
        yield {
            "image_id": image_id,
            "video_id": video_id,
            "track_id": track_ids[key],
            "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
            "score": score,
            # TrackEval TAO-OW ignores category IDs and assigns class 1 to all
            # GT and tracker detections.  This is deliberately not the native
            # category field above.
            "category_id": 1,
            "physical_track_id": physical_id,
            "_native_category_id": row.get("category_id"),
        }


def _build_tracker_json(native_path: Path, tracker_json: Path, spool_path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(str(spool_path))
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        "CREATE TABLE observations (video_id INTEGER, image_id INTEGER, physical_track_id TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL, score REAL, native_category_id INTEGER, input_order INTEGER, PRIMARY KEY(video_id, physical_track_id, image_id))"
    )
    connection.execute("CREATE INDEX observations_video_image ON observations(video_id, image_id)")
    try:
        # The generator inserts into SQLite as the JSON array is streamed.
        count, videos, tracks, categories = _atomic_json_array(
            tracker_json, _iter_tracker_rows(native_path, connection)
        )
        connection.commit()
        return {
            "rows": count,
            "videos": videos,
            "physical_tracks": tracks,
            "native_category_counts": {str(key): int(value) for key, value in sorted(categories.items())},
            "native_category_ids": sorted(int(key) for key in categories),
        }
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _atomic_native_teta_json(native_path: Path, target: Path) -> tuple[int, set[int]]:
    """Write a TAO JSON retaining native categories for the separate TETA run."""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    track_ids: dict[tuple[int, str], int] = {}
    categories: set[int] = set()
    next_track_id = 1
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("[")
            first = True
            for row_number, row in enumerate(_jsonl(native_path), 1):
                required = {"video_id", "image_id", "physical_track_id", "bbox_xyxy", "score", "category_id"}
                missing = required - set(row)
                if missing:
                    raise ValueError(f"native row {row_number} missing TETA fields: {sorted(missing)}")
                category = row.get("category_id")
                if category is None:
                    raise ValueError(f"native row {row_number} has no category_id for TETA")
                category_id = int(category)
                video_id = int(row["video_id"])
                image_id = int(row["image_id"])
                physical_id = str(row["physical_track_id"])
                box = [float(value) for value in row["bbox_xyxy"]]
                score = float(row["score"])
                if len(box) != 4 or not all(math.isfinite(value) for value in box) or not math.isfinite(score):
                    raise ValueError(f"native row {row_number} has invalid values for TETA")
                key = (video_id, physical_id)
                if key not in track_ids:
                    track_ids[key] = next_track_id
                    next_track_id += 1
                serialized = {
                    "image_id": image_id,
                    "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                    "score": score,
                    "category_id": category_id,
                    "video_id": video_id,
                    "track_id": track_ids[key],
                }
                if not first:
                    handle.write(",")
                first = False
                handle.write(json.dumps(serialized, sort_keys=True, separators=(",", ":"), allow_nan=False))
                categories.add(category_id)
                count += 1
            handle.write("]")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return count, categories


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else None,
    }


def _observability(
    spool_path: Path,
    gt_by_video: dict[int, list[dict[str, Any]]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Match frontend physical tracks to GT tracks and score observability."""

    connection = sqlite3.connect(str(spool_path))
    reliable: dict[str, bool] = {}
    matched_scores: dict[str, float] = {}
    matched_prediction_keys: set[tuple[int, str]] = set()
    candidate_frame_overlap = 0
    gt_count = 0
    try:
        for video_id in sorted(gt_by_video):
            gt_rows = gt_by_video[video_id]
            gt_count += len(gt_rows)
            predicted_by_frame: collections.defaultdict[int, list[tuple[str, list[float]]]] = collections.defaultdict(list)
            predicted_lengths: collections.Counter[str] = collections.Counter()
            for image_id, physical_id, x1, y1, x2, y2 in connection.execute(
                "SELECT image_id, physical_track_id, x1, y1, x2, y2 FROM observations WHERE video_id = ? ORDER BY image_id, input_order",
                (video_id,),
            ):
                physical_id = str(physical_id)
                predicted_by_frame[int(image_id)].append(
                    (physical_id, [float(x1), float(y1), float(x2), float(y2)])
                )
                predicted_lengths[physical_id] += 1
            prediction_ids = list(predicted_lengths)
            prediction_index = {key: index for index, key in enumerate(prediction_ids)}
            scores = np.zeros((len(gt_rows), len(prediction_ids)), dtype=np.float64)
            for gt_index, gt in enumerate(gt_rows):
                sums: collections.defaultdict[str, float] = collections.defaultdict(float)
                common: collections.Counter[str] = collections.Counter()
                for image_id, gt_box in gt["boxes"].items():
                    for physical_id, pred_box in predicted_by_frame.get(int(image_id), []):
                        common[physical_id] += 1
                        sums[physical_id] += _box_iou(gt_box, pred_box)
                candidate_frame_overlap += len(sums)
                for physical_id, value in sums.items():
                    denominator = len(gt["boxes"]) + predicted_lengths[physical_id] - common[physical_id]
                    if denominator > 0:
                        scores[gt_index, prediction_index[physical_id]] = value / denominator
            if scores.size:
                gt_indices, pred_indices = linear_sum_assignment(-scores)
                for gt_index, pred_index in zip(gt_indices, pred_indices):
                    gt = gt_rows[int(gt_index)]
                    value = float(scores[int(gt_index), int(pred_index)])
                    reliable[gt["sample_key"]] = value >= threshold
                    matched_scores[gt["sample_key"]] = value
                    matched_prediction_keys.add((video_id, prediction_ids[int(pred_index)]))
            for gt in gt_rows:
                reliable.setdefault(gt["sample_key"], False)
                matched_scores.setdefault(gt["sample_key"], 0.0)
    finally:
        connection.close()

    ordered = [row for rows in gt_by_video.values() for row in rows]
    ordered.sort(key=lambda row: int(row["stream_order"]))
    novel_rows = [row for row in ordered if row["gt_split"] == "new"]
    novel_reliable = sum(bool(reliable[row["sample_key"]]) for row in novel_rows)

    by_category: collections.defaultdict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in novel_rows:
        by_category[int(row["gt_category_id"])].append(row)
    persistent_categories = {
        category
        for category, rows in by_category.items()
        if len(rows) >= 2 and len({int(row["video_id"]) for row in rows}) >= 2
    }
    eligible_targets = 0
    reliable_targets = 0
    jointly_reliable = 0
    eligible_by_category: collections.Counter[int] = collections.Counter()
    for category, rows in by_category.items():
        previous: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda item: int(item["stream_order"])):
            prior_other_video = [item for item in previous if int(item["video_id"]) != int(row["video_id"])]
            if prior_other_video:
                eligible_targets += 1
                eligible_by_category[category] += 1
                target_reliable = bool(reliable[row["sample_key"]])
                source_reliable = any(bool(reliable[item["sample_key"]]) for item in prior_other_video)
                reliable_targets += int(target_reliable)
                jointly_reliable += int(target_reliable and source_reliable)
            previous.append(row)

    split_counts: dict[str, dict[str, int]] = {}
    for split in ("old", "new"):
        rows = [row for row in ordered if row["gt_split"] == split]
        split_counts[split] = {
            "tracks": len(rows),
            "reliably_observed": sum(bool(reliable[row["sample_key"]]) for row in rows),
        }
    return {
        "status": "COMPLETE",
        "threshold": float(threshold),
        "reliable_rule": "one-to-one GT/frontend track matching by temporal bbox IoU; reliable iff temporal IoU >= 0.5",
        "temporal_iou_definition": "sum of same-image box IoUs divided by the union of GT and predicted frame sets",
        "gt_tracks_scored": gt_count,
        "matched_gt_tracks": sum(1 for value in matched_scores.values() if value > 0.0),
        "matched_prediction_tracks": len(matched_prediction_keys),
        "candidate_track_pairs_with_shared_frame": candidate_frame_overlap,
        "by_gt_split": split_counts,
        "novel_track_observability": _metric(novel_reliable, len(novel_rows)),
        "persistent_observability": _metric(jointly_reliable, eligible_targets),
        "persistent_target_observability": _metric(reliable_targets, eligible_targets),
        "persistent_reuse_population": {
            "genuine_novel_categories": len(by_category),
            "persistent_categories": len(persistent_categories),
            "persistent_category_ids": sorted(persistent_categories),
            "eligible_target_tracks": eligible_targets,
            "eligible_targets_by_category": {str(key): int(value) for key, value in sorted(eligible_by_category.items())},
            "definition": "a genuine-novel GT track is eligible when an earlier stream track of the same category exists in another video; numerator additionally requires reliable source and target frontend matches",
        },
        "matched_temporal_iou_by_gt_track": {
            key: float(value) for key, value in sorted(matched_scores.items())
        },
    }


def _run_trackeval(
    *,
    tracker_json: Path,
    run_root: Path,
    tracker_name: str,
    gt_path: Path = TRACK_EVAL_GT,
) -> dict[str, Any]:
    """Run the pinned single-process class-agnostic TAO-OW evaluator."""

    if not gt_path.is_file():
        raise FileNotFoundError(gt_path)
    # TrackEval's vendored code still references aliases removed by NumPy 2;
    # keep the compatibility shim process-local and do not edit third_party.
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    sys.path.insert(0, str(ROOT / "third_party/TrackEval"))
    import trackeval  # noqa: WPS433, E402

    trackers_root = run_root / "trackers"
    tracker_data = trackers_root / tracker_name / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)
    tracker_link = tracker_data / "tao_track.json"
    if tracker_link.exists() or tracker_link.is_symlink():
        raise FileExistsError(tracker_link)
    temporary_link = tracker_link.with_name(f".{tracker_link.name}.tmp.{os.getpid()}")
    os.symlink(tracker_json.resolve(), temporary_link)
    os.replace(temporary_link, tracker_link)

    result_root = run_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": False,
        "PLOT_CURVES": False,
    })
    dataset_config = trackeval.datasets.TAO_OW.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(gt_path.parent),
        "TRACKERS_FOLDER": str(trackers_root),
        "OUTPUT_FOLDER": str(result_root),
        "TRACKERS_TO_EVAL": [tracker_name],
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "MAX_DETECTIONS": 300,
        "SUBSET": "all",
        "PRINT_CONFIG": False,
    })
    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.TAO_OW(dataset_config)
    result, messages = evaluator.evaluate([dataset], [trackeval.metrics.HOTA()])
    summary_path = result_root / tracker_name / "cls_comb_det_av_summary.txt"
    if not summary_path.is_file():
        raise FileNotFoundError(f"TrackEval did not write combined summary: {summary_path}")
    lines = [line.split() for line in summary_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"invalid TrackEval summary: {summary_path}")
    fields, values = lines[-2], lines[-1]
    parsed: dict[str, float] = {}
    for field, value in zip(fields, values):
        try:
            parsed[field] = float(value)
        except ValueError:
            continue
    # TrackEval summaries print float metrics in percentage units.  Keep both
    # units explicit so they cannot be confused with TrackOCD [0, 1] values.
    percent_fields = ("HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA", "OWTA")
    normalized = {field: float(parsed[field] / 100.0) for field in percent_fields if field in parsed}
    return {
        "status": "COMPLETE",
        "evaluator": "vendored TrackEval TAO_OW",
        "evaluator_source": str((ROOT / "third_party/TrackEval").resolve()),
        "gt_path": str(gt_path.resolve()),
        "gt_sha256": sha256_file(gt_path),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256_file(summary_path),
        "tracker_json": str(tracker_json.resolve()),
        "tracker_json_sha256": sha256_file(tracker_json),
        "config": {
            "subset": "all",
            "max_detections_per_image": 300,
            "parallel": False,
            "cores": 1,
            "metric": "HOTA family; TAO-OW class-agnostic single foreground class",
        },
        "values_percent": {field: float(parsed[field]) for field in percent_fields if field in parsed},
        "values": normalized,
        "result_type": type(result).__name__,
        "messages": messages,
    }


def _run_teta_reference(
    *,
    native_path: Path,
    run_root: Path,
    tracker_name: str,
    annotation_path: Path = TETA_ANNOTATION,
    known_ids: Path = KNOWN_IDS,
    novel_ids: Path = NOVEL_IDS,
) -> dict[str, Any]:
    """Run one serialized category-aware TETA reference subprocess."""

    if not TETA_PYTHON.is_file():
        raise FileNotFoundError(TETA_PYTHON)
    for path in (annotation_path, known_ids, novel_ids):
        if not path.is_file():
            raise FileNotFoundError(path)
    teta_root = run_root / "teta"
    teta_input = teta_root / "native_input" / "tao_track.json"
    _atomic_native_teta_json(native_path, teta_input)
    output_json = teta_root / "teta_reference.json"
    command = [
        str(TETA_PYTHON),
        str((ROOT / "scripts/trackocd_v2/run_teta_reference.py").resolve()),
        "--tracker-json", str(teta_input.resolve()),
        "--annotation", str(annotation_path.resolve()),
        "--output-root", str(teta_root.resolve()),
        "--output-json", str(output_json.resolve()),
        "--tracker-name", tracker_name,
        "--known-ids", str(known_ids.resolve()),
        "--novel-ids", str(novel_ids.resolve()),
    ]
    process = subprocess.run(command, cwd=ROOT, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"TETA reference failed with return code {process.returncode}")
    if not output_json.is_file():
        raise FileNotFoundError(output_json)
    result = json.loads(output_json.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"TETA reference audit is not complete: {result.get('status')}")
    result["command"] = command
    result["input_native_rows"] = str(native_path.resolve())
    result["input_native_rows_sha256"] = sha256_file(native_path)
    return result


def _stage_path(frontend_slug: str, *, split: str = "val") -> Path:
    suffix = "" if split == "val" else "_test"
    return OUTPUT_TARGET / "audit" / f"frontend_{frontend_slug}{suffix}.json"


def run(
    frontend_slug: str,
    *,
    run_id: str | None = None,
    skip_trackeval: bool = False,
    split: str = "val",
) -> dict[str, Any]:
    if frontend_slug not in STAGE_NAMES:
        raise ValueError(f"unknown frontend slug: {frontend_slug}")
    if split not in {"val", "test"}:
        raise ValueError(f"unknown frontend metrics split: {split}")
    # This check intentionally precedes the Test stage read and all Test
    # annotation/label access.  Val metrics remain available before freeze.
    if split == "test":
        assert_test_semantic_access_allowed(
            OUTPUT_TARGET / "audit/FINAL_FREEZE.json",
            "run frozen TAO Test frontend metrics",
        )
    frontend = STAGE_NAMES[frontend_slug]
    route = route_specs(ROOT)[frontend]
    out = ensure_output_layout()
    stage_path = _stage_path(frontend_slug, split=split)
    if not stage_path.is_file():
        raise FileNotFoundError(stage_path)
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    required_status = "FINAL_TEST_NATIVE_STREAM_NORMALIZED" if split == "test" else None
    if required_status is not None and stage.get("status") != required_status:
        raise RuntimeError(f"frontend Test stage is not finalized: {stage_path}")
    if stage.get("native_run_complete") is not True or stage.get("same_v2_evaluator_contract") is not True:
        raise RuntimeError(f"frontend stage is not a completed native v2 stream: {stage_path}")
    if stage.get("physical_stream_contract_complete") is not True:
        raise RuntimeError(f"frontend stage lacks physical stream contract: {stage_path}")
    if split == "test" and stage.get("same_v2_physical_metric_protocol") is not True:
        raise RuntimeError(f"frontend Test stage lacks the shared physical metric protocol: {stage_path}")
    normalized = stage.get("normalized_outputs") or {}
    native_path = Path(str(normalized.get("native_evaluator_rows", "")))
    physical_path = Path(str(normalized.get("physical_stream", "")))
    if not native_path.is_file() or not physical_path.is_file():
        raise FileNotFoundError(f"normalized outputs missing: {native_path}, {physical_path}")
    forbidden = {"category_id", "category_name", "text", "gt_category_id", "gt_split", "gt_match_id"}
    for index, row in enumerate(_jsonl(physical_path)):
        if forbidden & set(row):
            raise ValueError(f"physical stream contains forbidden evaluator/model field at row {index}")
        if index >= 9:
            break

    gt_manifest = GT_MANIFEST
    gt_labels = GT_LABELS
    track_eval_gt = TRACK_EVAL_GT
    teta_annotation = TETA_ANNOTATION
    if split == "test":
        gt_manifest = TEST_MANIFEST
        gt_labels = TEST_LABELS
        track_eval_gt = TEST_TRACK_EVAL_GT
        teta_annotation = TEST_TETA_ANNOTATION
        if not gt_manifest.is_file() or not gt_labels.is_file():
            build_manifest = subprocess.run(
                [sys.executable, str((ROOT / "scripts/trackocd_v2/build_test_track_stream.py").resolve()), "--final"],
                cwd=ROOT,
                check=False,
            )
            if build_manifest.returncode != 0:
                raise RuntimeError(f"final Test GT manifest build failed: {build_manifest.returncode}")
        if not track_eval_gt.is_file():
            build_trackeval = subprocess.run(
                [sys.executable, str((ROOT / "scripts/trackocd_v2/build_test_trackeval_gt.py").resolve())],
                cwd=ROOT,
                check=False,
            )
            if build_trackeval.returncode != 0:
                raise RuntimeError(f"Test TrackEval GT build failed: {build_trackeval.returncode}")
    gt_by_video, _ = _load_gt_tracks(gt_manifest, gt_labels)
    if run_id is None:
        run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    run_root = out / "metrics" / "frontend" / frontend_slug / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    tracker_json = run_root / "tracker_input" / "tao_ow_track.json"
    spool_path = run_root / "observability_spool.sqlite"
    try:
        conversion = _build_tracker_json(native_path, tracker_json, spool_path)
        observability = _observability(spool_path, gt_by_video)
        tracking = None if skip_trackeval else _run_trackeval(
            tracker_json=tracker_json,
            run_root=run_root / "trackeval",
            tracker_name=frontend_slug,
            gt_path=track_eval_gt,
        )
    finally:
        if spool_path.exists():
            spool_path.unlink()

    category_ids = set(conversion["native_category_ids"])
    if category_ids == {1}:
        teta = {
            "status": "NOT_APPLICABLE_CATEGORY_AGNOSTIC_ROUTE",
            "value": None,
            "reason": "native stream exports only the SimOWT-style foreground class 1; category-aware TETA would measure an absent semantic class, so no TETA number is fabricated",
        }
    else:
        try:
            teta = _run_teta_reference(
                native_path=native_path,
                run_root=run_root,
                tracker_name=frontend_slug,
                annotation_path=teta_annotation,
                known_ids=KNOWN_IDS,
                novel_ids=NOVEL_IDS,
            )
        except Exception as exc:
            if frontend_slug != "ovtr":
                raise
            teta = _record_ovtr_teta_failure(
                exc=exc,
                run_root=run_root,
                native_path=native_path,
                split=split,
            )
    result = {
        "schema_version": "trackocd.v2.frontend_metrics.v1",
        "status": "COMPLETE" if tracking is not None else "OBSERVABILITY_COMPLETE_TRACKING_PENDING",
        "generated_utc": _now(),
        "frontend": frontend,
        "frontend_slug": frontend_slug,
        "frontend_role": route["frontend_role"],
        "candidate_for_final_frontend": bool(route["candidate_for_final_frontend"]),
        "clean_trackocd_final_frontend": bool(route["clean_trackocd_final_frontend"]),
        "split": split,
        "stage_asset": str(stage_path.resolve()),
        "stage_asset_sha256": sha256_file(stage_path),
        "native_stream": {
            "path": str(native_path.resolve()),
            "sha256": sha256_file(native_path),
            "physical_stream_path": str(physical_path.resolve()),
            "physical_stream_sha256": sha256_file(physical_path),
            "conversion": conversion,
        },
        "tracking": tracking,
        "teta_native_reference": teta,
        "observability": observability,
        "metric_contract": {
            "category_free_physical_metrics": ["OWTA", "AssA", "DetRe", "LocA"],
            "native_category_aware_reference": ["TETA"],
            "native_category_aware_reference_role": route.get("native_teta_role", "reference_only"),
            "gt_join_used_for_model_or_native_stream": False,
            "gt_join_used_for_scoring_only": True,
            "test_semantic_accessed": split == "test",
            "category_free_tracker_input_category_id": 1,
        },
        "run_root": str(run_root.resolve()),
        "run_id": run_id,
        "skip_trackeval": bool(skip_trackeval),
        "test_semantic_accessed": split == "test",
        "test_selection_or_tuning": False,
    }
    audit_suffix = "" if split == "val" else "_test"
    audit_path = out / "audit" / f"frontend_{frontend_slug}_metrics{audit_suffix}.json"
    atomic_json(audit_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=tuple(STAGE_NAMES), required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--skip-trackeval", action="store_true", help="only for bounded smoke/debug runs; does not produce complete metrics")
    args = parser.parse_args()
    try:
        result = run(args.frontend, run_id=args.run_id, skip_trackeval=args.skip_trackeval, split=args.split)
    except Exception as exc:
        out = ensure_output_layout()
        failure = {
            "schema_version": "trackocd.v2.frontend_metrics.v1",
            "status": "FAILED_FRONTEND_METRICS",
            "generated_utc": _now(),
            "frontend_slug": args.frontend,
            "split": args.split,
            "error": f"{type(exc).__name__}: {exc}",
            "test_semantic_accessed": args.split == "test",
        }
        audit_suffix = "" if args.split == "val" else "_test"
        atomic_json(out / "audit" / f"frontend_{args.frontend}_metrics{audit_suffix}.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
