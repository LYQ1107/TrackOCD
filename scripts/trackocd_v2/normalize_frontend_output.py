#!/usr/bin/env python3
"""Normalize one native frontend output into the TrackOCD v2 stream contract.

Two input forms are supported:

* ``grouped_jsonl``: one row per track, as used by the existing public
  SimOWT/Q0 export;
* ``tao_json``: a JSON array of per-frame TAO rows, as emitted by OVTR and
  COVTrack.

The native category is retained only in the separate evaluator-row output.
The physical stream passed to TrackOCD never contains category, text, GT, or
model semantic fields.  A SQLite spool keeps the conversion bounded in RAM;
the spool is temporary and is removed after the two atomic output files are
closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.frontend_contract import FRONTENDS, NATIVE_ROW_FIELDS, PHYSICAL_FIELDS, route_specs  # noqa: E402
from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


ANNOTATION = ROOT / "data/raw/tao/annotations/validation.json"
FORMAT_NAMES = ("auto", "grouped_jsonl", "tao_json")


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    """Yield objects from a top-level JSON array without loading the array."""

    decoder = json.JSONDecoder()
    chunk_size = 1 << 20
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def more() -> None:
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def skip_space() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                more()

        more()
        skip_space()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"expected a top-level JSON array: {path}")
        position += 1

        while True:
            skip_space()
            if position >= len(buffer):
                raise ValueError(f"unterminated JSON array: {path}")
            if buffer[position] == "]":
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"invalid or truncated JSON array: {path}")
                    buffer = buffer[position:]
                    position = 0
                    more()
            if not isinstance(value, dict):
                raise ValueError(f"top-level JSON array contains a non-object: {path}")
            yield value
            position = end
            skip_space()
            while position >= len(buffer) and not eof:
                more()
                skip_space()
            if position >= len(buffer):
                raise ValueError(f"unterminated JSON array: {path}")
            delimiter = buffer[position]
            if delimiter == ",":
                position += 1
            elif delimiter == "]":
                return
            else:
                raise ValueError(f"expected ',' or ']' in JSON array: {path}")
            if position >= (1 << 20):
                buffer = buffer[position:]
                position = 0


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object: {path}")
            yield value


def _detect_format(path: Path) -> str:
    if path.suffix.lower() == ".jsonl":
        return "grouped_jsonl"
    return "tao_json"


def _load_image_index(path: Path) -> tuple[dict[int, dict[str, Any]], str]:
    """Load only structural image metadata; annotation labels are not read."""

    value = json.loads(path.read_text(encoding="utf-8"))
    images: dict[int, dict[str, Any]] = {}
    for row in value.get("images", []):
        image_id = int(row["id"])
        images[image_id] = {
            "video_id": int(row["video_id"]),
            "frame_index": int(row.get("frame_index", image_id)),
            "file_name": str(row["file_name"]),
        }
    return images, _sha256(path)


def _float_box(box: Any, *, xywh: bool, context: str) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"box must have four coordinates at {context}")
    values = [float(item) for item in box]
    if xywh:
        values = [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    return values


def _finite_non_degenerate(box: list[float]) -> bool:
    return all(math.isfinite(value) for value in box) and box[2] > box[0] and box[3] > box[1]


def _as_score(value: Any, default: float = 1.0) -> float:
    if value is None:
        return default
    return float(value)


def _track_id(row: dict[str, Any]) -> str:
    for key in ("physical_track_id", "track_id", "id"):
        if key in row:
            return str(row[key])
    raise ValueError("frontend row has no track_id/physical_track_id")


def _insert_grouped(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    input_order: int,
    images: dict[int, dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    video_id = int(row["video_id"])
    physical_id = _track_id(row)
    sample_id = str(row.get("source_sample_id", row.get("sample_id", f"{video_id}_{physical_id}")))
    frame_ids = [int(value) for value in row.get("frame_ids", [])]
    boxes = row.get("boxes_xyxy")
    if not isinstance(boxes, list) or len(frame_ids) != len(boxes):
        raise ValueError(f"grouped row has mismatched frame_ids/boxes: {sample_id}")
    paths = row.get("image_paths") or [None] * len(frame_ids)
    scores = row.get("scores")
    quality = row.get("quality")
    if scores is None:
        scores = quality if quality is not None else [1.0] * len(frame_ids)
    if len(paths) != len(frame_ids) or len(scores) != len(frame_ids):
        raise ValueError(f"grouped row has mismatched observation arrays: {sample_id}")
    category_ids = row.get("category_ids")
    if category_ids is not None and len(category_ids) != len(frame_ids):
        raise ValueError(f"grouped row has mismatched category_ids: {sample_id}")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"duplicate frame in grouped track: {sample_id}")
    for index, (frame_id, box, source_path, score) in enumerate(zip(frame_ids, boxes, paths, scores)):
        context = f"{sample_id}[{index}]"
        xyxy = _float_box(box, xywh=False, context=context)
        stats["observations_seen"] += 1
        if not _finite_non_degenerate(xyxy):
            stats["invalid_boxes"] += 1
            if len(stats["invalid_examples"]) < 5:
                stats["invalid_examples"].append(context)
        image = images.get(frame_id)
        if image is not None and int(image["video_id"]) != video_id:
            raise ValueError(f"image/video mismatch at {context}: {video_id} vs {image['video_id']}")
        frame_index = int(image["frame_index"]) if image is not None else frame_id
        file_name = str(source_path) if source_path is not None else (str(image["file_name"]) if image else None)
        category = None if category_ids is None else int(category_ids[index])
        try:
            conn.execute(
                "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (video_id, physical_id, frame_id, frame_index, file_name, *xyxy, _as_score(score), category, sample_id, input_order),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"duplicate (video,track,image) at {context}") from exc


def _insert_tao(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    input_order: int,
    images: dict[int, dict[str, Any]],
    stats: dict[str, Any],
) -> None:
    if "image_id" not in row or "video_id" not in row:
        raise ValueError("TAO row requires image_id and video_id")
    video_id = int(row["video_id"])
    image_id = int(row["image_id"])
    physical_id = _track_id(row)
    image = images.get(image_id)
    if image is None:
        raise ValueError(f"TAO row image_id is absent from validation metadata: {image_id}")
    if int(image["video_id"]) != video_id:
        raise ValueError(f"TAO row image/video mismatch: {image_id}")
    xyxy = _float_box(row.get("bbox"), xywh=True, context=f"{video_id}/{physical_id}/{image_id}")
    stats["observations_seen"] += 1
    if not _finite_non_degenerate(xyxy):
        stats["invalid_boxes"] += 1
        if len(stats["invalid_examples"]) < 5:
            stats["invalid_examples"].append(f"{video_id}/{physical_id}/{image_id}")
    category = row.get("category_id")
    category_value = None if category is None else int(category)
    try:
        conn.execute(
            "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (video_id, physical_id, image_id, int(image["frame_index"]), str(image["file_name"]), *xyxy, _as_score(row.get("score")), category_value, f"{video_id}_{physical_id}", input_order),
        )
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"duplicate (video,track,image) at input row {input_order}") from exc


def _jsonl_handle(path: Path) -> tuple[Path, Any]:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    return temporary, temporary.open("w", encoding="utf-8")


def normalize_file(
    *,
    frontend: str,
    input_path: Path,
    input_format: str,
    physical_output: Path,
    native_output: Path,
    audit_output: Path,
    annotation_path: Path | None = ANNOTATION,
    allow_invalid_boxes: bool = False,
    source_split: str = "val_predicted",
) -> dict[str, Any]:
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown frontend: {frontend}")
    if source_split not in {"val_predicted", "test_predicted"}:
        raise ValueError(f"unsupported source split: {source_split}")
    route = route_specs(ROOT)[frontend]
    frontend_slug = str(route["slug"])
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_format == "auto":
        input_format = _detect_format(input_path)
    if input_format not in FORMAT_NAMES[1:]:
        raise ValueError(f"unsupported input format: {input_format}")
    images: dict[int, dict[str, Any]] = {}
    annotation_sha256 = None
    if input_format == "tao_json":
        if annotation_path is None or not annotation_path.is_file():
            raise FileNotFoundError(annotation_path or ANNOTATION)
        images, annotation_sha256 = _load_image_index(annotation_path)

    stats: dict[str, Any] = {
        "tracks": 0,
        "observations_seen": 0,
        "videos": set(),
        "invalid_boxes": 0,
        "invalid_examples": [],
        "input_rows": 0,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    spool_path = Path(tempfile.mkstemp(prefix="trackocd_frontend_", suffix=".sqlite", dir=str(audit_output.parent))[1])
    conn = sqlite3.connect(str(spool_path))
    try:
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute(
            "CREATE TABLE observations (video_id INTEGER, physical_track_id TEXT, image_id INTEGER, frame_index INTEGER, file_name TEXT, x1 REAL, y1 REAL, x2 REAL, y2 REAL, score REAL, category_id INTEGER, source_sample_id TEXT, input_order INTEGER, PRIMARY KEY(video_id, physical_track_id, image_id))"
        )
        rows: Iterable[dict[str, Any]] = _iter_jsonl(input_path) if input_format == "grouped_jsonl" else _iter_json_array(input_path)
        for input_order, row in enumerate(rows):
            stats["input_rows"] += 1
            if input_format == "grouped_jsonl":
                _insert_grouped(conn, row, input_order, images, stats)
            else:
                _insert_tao(conn, row, input_order, images, stats)
            if input_order % 5000 == 0:
                conn.commit()
        conn.commit()
        if stats["invalid_boxes"] and not allow_invalid_boxes:
            raise ValueError(f"invalid or degenerate boxes: {stats['invalid_boxes']}")

        physical_tmp, physical_handle = _jsonl_handle(physical_output)
        native_tmp, native_handle = _jsonl_handle(native_output)
        try:
            query = """
                SELECT o.video_id, o.physical_track_id, o.image_id, o.frame_index,
                       o.file_name, o.x1, o.y1, o.x2, o.y2, o.score,
                       o.category_id, o.source_sample_id
                FROM observations AS o
                JOIN (
                    SELECT video_id, physical_track_id,
                           MIN(frame_index) AS first_frame,
                           MIN(image_id) AS first_image,
                           MIN(input_order) AS first_order
                    FROM observations
                    GROUP BY video_id, physical_track_id
                ) AS firsts
                  ON firsts.video_id = o.video_id
                 AND firsts.physical_track_id = o.physical_track_id
                ORDER BY firsts.first_frame, firsts.video_id, firsts.first_image,
                         firsts.first_order, firsts.physical_track_id,
                         o.frame_index, o.image_id
            """
            previous_key: tuple[int, str] | None = None
            physical_record: dict[str, Any] | None = None
            stream_order = 0
            for row in conn.execute(query):
                video_id, physical_id, image_id, frame_index, file_name, x1, y1, x2, y2, score, category, source_sample_id = row
                key = (int(video_id), str(physical_id))
                if previous_key != key:
                    if physical_record is not None:
                        physical_handle.write(json.dumps(physical_record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                    physical_record = {
                        "sample_key": f"{frontend_slug}_{video_id}_{physical_id}",
                        "source_sample_id": str(source_sample_id),
                        "video_id": int(video_id),
                        "physical_track_id": str(physical_id),
                        "frame_ids": [],
                        "boxes_xyxy": [],
                        "image_paths": [],
                        "quality": [],
                        "stream_order": stream_order,
                        "source_split": source_split,
                    }
                    stream_order += 1
                    stats["tracks"] += 1
                    stats["videos"].add(int(video_id))
                    previous_key = key
                assert physical_record is not None
                box = [float(x1), float(y1), float(x2), float(y2)]
                physical_record["frame_ids"].append(int(image_id))
                physical_record["boxes_xyxy"].append(box)
                physical_record["image_paths"].append(str(file_name) if file_name is not None else "")
                physical_record["quality"].append(max(float(score), 1e-6))
                native_handle.write(json.dumps({
                    "video_id": int(video_id),
                    "image_id": int(image_id),
                    "frame_index": int(frame_index),
                    "physical_track_id": str(physical_id),
                    "bbox_xyxy": box,
                    "score": float(score),
                    "category_id": None if category is None else int(category),
                }, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            if physical_record is not None:
                physical_handle.write(json.dumps(physical_record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            physical_handle.flush()
            native_handle.flush()
            os.fsync(physical_handle.fileno())
            os.fsync(native_handle.fileno())
        finally:
            physical_handle.close()
            native_handle.close()
        os.replace(physical_tmp, physical_output)
        os.replace(native_tmp, native_output)
    except Exception:
        for path in (locals().get("physical_tmp"), locals().get("native_tmp")):
            if isinstance(path, Path) and path.exists():
                path.unlink()
        raise
    finally:
        conn.close()
        if spool_path.exists():
            spool_path.unlink()

    result = {
        "schema_version": "trackocd.v2.frontend_normalization.v1",
        "status": "COMPLETE_WITH_INVALID_BOXES" if stats["invalid_boxes"] else "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frontend": frontend,
        "frontend_slug": frontend_slug,
        "source_split": source_split,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "format": input_format,
            "input_rows": int(stats["input_rows"]),
        },
        "annotation": {
            "path": str(annotation_path.resolve()) if annotation_path is not None else None,
            "sha256": annotation_sha256,
            "used_for": "structural image/video/frame_index/file_name lookup only",
            "semantic_labels_consumed": False,
        },
        "outputs": {
            "physical_stream": str(physical_output.resolve()),
            "physical_stream_sha256": _sha256(physical_output),
            "native_evaluator_rows": str(native_output.resolve()),
            "native_evaluator_rows_sha256": _sha256(native_output),
        },
        "counts": {
            "tracks": int(stats["tracks"]),
            "observations": int(stats["observations_seen"]),
            "videos": len(stats["videos"]),
            "invalid_boxes": int(stats["invalid_boxes"]),
        },
        "invalid_box_policy": {
            "allow_invalid_boxes": bool(allow_invalid_boxes),
            "invalid_examples": list(stats["invalid_examples"]),
            "all_observations_retained": True,
        },
        "contracts": {
            "physical_stream_fields": list(PHYSICAL_FIELDS),
            "native_evaluator_row_fields": list(NATIVE_ROW_FIELDS),
            "physical_stream_contains_category_or_text": False,
            "gt_join_used_for_normalization": False,
            "test_semantic_accessed": False,
        },
    }
    atomic_json(audit_output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=FRONTENDS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-format", choices=FORMAT_NAMES, default="auto")
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--physical-output", type=Path, default=None)
    parser.add_argument("--native-output", type=Path, default=None)
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--allow-invalid-boxes", action="store_true", help="retain and audit invalid/degenerate boxes instead of failing; never drops them")
    parser.add_argument("--source-split", choices=("val_predicted", "test_predicted"), default="val_predicted")
    args = parser.parse_args()
    out = ensure_output_layout()
    slug = route_specs()[args.frontend]["slug"]
    output_dir = args.output_dir or (out / "manifests/frontend_streams" / slug)
    physical_output = args.physical_output or (output_dir / "physical_tracks.jsonl")
    native_output = args.native_output or (output_dir / "native_evaluator_rows.jsonl")
    audit_output = args.audit_output or (out / "audit" / f"frontend_{slug}_normalization.json")
    try:
        result = normalize_file(
            frontend=args.frontend,
            input_path=args.input.resolve(),
            input_format=args.input_format,
            physical_output=physical_output.resolve(),
            native_output=native_output.resolve(),
            audit_output=audit_output.resolve(),
            annotation_path=args.annotation.resolve() if args.annotation is not None else None,
            allow_invalid_boxes=args.allow_invalid_boxes,
            source_split=args.source_split,
        )
    except Exception as exc:
        failure = {
            "schema_version": "trackocd.v2.frontend_normalization.v1",
            "status": "FAILED_NORMALIZATION",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "frontend": args.frontend,
            "input": str(args.input.resolve()),
            "error": f"{type(exc).__name__}: {exc}",
            "outputs_committed": False,
            "allow_invalid_boxes": bool(args.allow_invalid_boxes),
            "test_semantic_accessed": False,
        }
        atomic_json(audit_output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
