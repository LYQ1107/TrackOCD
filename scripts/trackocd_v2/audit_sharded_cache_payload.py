#!/usr/bin/env python3
"""Numerically audit completed TrackOCD v2 sharded feature payloads.

The audit is read-only and intentionally independent of GT/evaluator code. It
opens only the completed shard arrays, their Parquet lineage indexes, and
``.done`` markers. A partial cache is reported as reusable partial evidence;
it is not promoted to a formal cache until the manifest covers every track.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PREFIXES = (1, 2, 4, 8, 16)
FORBIDDEN_FIELDS = {
    "category_id",
    "category_name",
    "text",
    "gt_category_id",
    "gt_split",
    "gt_match_id",
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _cache_root(manifest_path: Path, payload: dict[str, Any]) -> Path:
    raw = payload.get("cache_root")
    if not raw:
        raise ValueError("manifest misses cache_root")
    root = Path(str(raw))
    return root.resolve() if root.is_absolute() else (manifest_path.parent / root).resolve()


def _finite_array(array: np.ndarray, chunk_rows: int = 8192) -> bool:
    for start in range(0, int(array.shape[0]), chunk_rows):
        if not np.isfinite(array[start : start + chunk_rows]).all():
            return False
    return True


def _check_boxes(row: dict[str, Any]) -> int:
    invalid = 0
    boxes = row.get("boxes_xyxy", [])
    for box in boxes:
        values = [float(value) for value in box]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            invalid += 1
        elif values[2] <= values[0] or values[3] <= values[1]:
            invalid += 1
    return invalid


def _audit_shard(
    shard_dir: Path,
    entry: dict[str, Any],
    expected_source_sha256: str,
    seen_keys: set[str],
    seen_physical_ids: set[tuple[int, str]],
    expected_global_start: int,
    previous_stream_order: int,
) -> tuple[dict[str, Any], list[str], int, int]:
    errors: list[str] = []
    done = _json(shard_dir / ".done")
    for key in ("track_start", "track_count", "observation_count"):
        if int(done.get(key, -1)) != int(entry.get(key, -2)):
            errors.append(f"{shard_dir.name}:done_{key}_mismatch")
    if done.get("source_sha256") != expected_source_sha256:
        errors.append(f"{shard_dir.name}:done_source_sha256_mismatch")

    index_rows = _read_index(shard_dir / "index.parquet")
    track_count = int(entry["track_count"])
    observation_count = int(entry["observation_count"])
    if len(index_rows) != track_count:
        errors.append(f"{shard_dir.name}:index_rows={len(index_rows)} expected={track_count}")

    observations = np.load(shard_dir / "observations.npy", mmap_mode="r", allow_pickle=False)
    prefixes = np.load(shard_dir / "prefix_features.npy", mmap_mode="r", allow_pickle=False)
    full = np.load(shard_dir / "full_features.npy", mmap_mode="r", allow_pickle=False)
    if observations.shape != (observation_count, 768):
        errors.append(f"{shard_dir.name}:observations_shape={observations.shape}")
    if prefixes.shape != (track_count, len(PREFIXES), 768):
        errors.append(f"{shard_dir.name}:prefix_features_shape={prefixes.shape}")
    if full.shape != (track_count, 768):
        errors.append(f"{shard_dir.name}:full_features_shape={full.shape}")
    for name, array in (("observations", observations), ("prefix_features", prefixes), ("full_features", full)):
        if array.dtype not in (np.float16, np.float32):
            errors.append(f"{shard_dir.name}:{name}_dtype={array.dtype}")
        elif not _finite_array(array):
            errors.append(f"{shard_dir.name}:{name}_nonfinite")

    expected_offset = 0
    last_order = previous_stream_order
    invalid_boxes = 0
    for position, row in enumerate(index_rows):
        key = str(row["sample_key"])
        if key in seen_keys:
            errors.append(f"{shard_dir.name}:duplicate_sample_key:{key}")
        seen_keys.add(key)
        global_index = int(row["global_track_index"])
        expected_index = expected_global_start + position
        if global_index != expected_index:
            errors.append(f"{shard_dir.name}:global_track_index={global_index} expected={expected_index}")
        source_split = str(row.get("source_split", ""))
        if source_split != "val_predicted":
            errors.append(f"{shard_dir.name}:source_split={source_split}")
        leaked = FORBIDDEN_FIELDS & set(row)
        if leaked:
            errors.append(f"{shard_dir.name}:forbidden_fields={sorted(leaked)}")
        physical_key = (int(row["video_id"]), str(row["physical_track_id"]))
        if physical_key in seen_physical_ids:
            errors.append(f"{shard_dir.name}:duplicate_physical_track={physical_key}")
        seen_physical_ids.add(physical_key)

        offset = int(row["observation_offset"])
        count = int(row["observation_count"])
        if offset != expected_offset or count < 1 or offset + count > observation_count:
            errors.append(f"{shard_dir.name}:observation_offset_or_count")
        expected_offset = offset + count
        stream_order = int(row["stream_order"])
        if stream_order < last_order:
            errors.append(f"{shard_dir.name}:stream_order_regression")
        last_order = stream_order
        lengths = [
            len(row.get("frame_ids", [])),
            len(row.get("image_paths", [])),
            len(row.get("boxes_xyxy", [])),
            len(row.get("quality", [])),
            len(row.get("prefix_observations_used", [])),
        ]
        if lengths[:4] != [count, count, count, count] or lengths[4] != len(PREFIXES):
            errors.append(f"{shard_dir.name}:lineage_lengths={lengths}")
        prefix_counts = [int(value) for value in row.get("prefix_observations_used", [])]
        if any(value < 1 or value > count for value in prefix_counts):
            errors.append(f"{shard_dir.name}:prefix_observation_count")
        if any(str(path).startswith("test/") or "/test/" in str(path) for path in row.get("image_paths", [])):
            errors.append(f"{shard_dir.name}:test_path")
        invalid_boxes += _check_boxes(row)
    if expected_offset != observation_count:
        errors.append(f"{shard_dir.name}:observation_coverage={expected_offset} expected={observation_count}")

    return (
        {
            "shard": shard_dir.name,
            "track_start": int(entry["track_start"]),
            "track_count": track_count,
            "observation_count": observation_count,
            "invalid_box_count": invalid_boxes,
        },
        errors,
        last_order,
        invalid_boxes,
    )


def _read_index(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for sharded cache audit") from exc
    return parquet.read_table(str(path)).to_pylist()


def audit(manifest_path: Path, expected_source_sha256: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    manifest_path = manifest_path.resolve()
    payload = _json(manifest_path)
    cache_root = _cache_root(manifest_path, payload)
    input_sha256 = str((payload.get("input") or {}).get("sha256") or "")
    expected_sha256 = expected_source_sha256 or input_sha256
    errors: list[str] = []
    if payload.get("format") != "sharded_numpy_with_parquet_index":
        errors.append("manifest:format_mismatch")
    if tuple(int(value) for value in payload.get("prefixes", [])) != PREFIXES:
        errors.append("manifest:prefix_contract_mismatch")
    if expected_source_sha256 is not None and input_sha256 != expected_source_sha256:
        errors.append("manifest:input_source_sha256_mismatch")
    entries = sorted(payload.get("completed_shards") or [], key=lambda item: int(item["track_start"]))
    seen_keys: set[str] = set()
    seen_physical_ids: set[tuple[int, str]] = set()
    reports: list[dict[str, Any]] = []
    total_tracks = total_observations = invalid_boxes = 0
    expected_global_start = 0
    previous_stream_order = -1
    for entry in entries:
        track_start = int(entry["track_start"])
        if track_start != expected_global_start:
            errors.append(f"manifest:track_range_gap_or_overlap_at={track_start}_expected={expected_global_start}")
        shard_index = track_start // int(payload["shard_tracks"])
        shard_dir = cache_root / "shards" / f"shard-{shard_index:06d}"
        try:
            report, shard_errors, previous_stream_order, shard_invalid_boxes = _audit_shard(
                shard_dir,
                entry,
                expected_sha256,
                seen_keys,
                seen_physical_ids,
                expected_global_start,
                previous_stream_order,
            )
        except Exception as exc:
            report = {"shard": shard_dir.name, "track_start": track_start}
            shard_errors = [f"{shard_dir.name}:exception:{type(exc).__name__}:{exc}"]
            shard_invalid_boxes = 0
        reports.append(report)
        errors.extend(shard_errors)
        total_tracks += int(entry["track_count"])
        total_observations += int(entry["observation_count"])
        invalid_boxes += shard_invalid_boxes
        expected_global_start += int(entry["track_count"])

    required_shards = (int(payload.get("total_tracks", 0)) + int(payload["shard_tracks"]) - 1) // int(payload["shard_tracks"])
    complete_coverage = (
        payload.get("status") == "COMPLETE"
        and len(entries) == required_shards
        and total_tracks == int(payload.get("total_tracks", -1))
        and not errors
    )
    status = "PASS_COMPLETE" if complete_coverage else ("PASS_PARTIAL_COMPLETED_SHARDS" if not errors else "FAIL")
    return {
        "schema_version": "trackocd.v2.sharded_cache_payload_audit.v1",
        "status": status,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
        "frontend": payload.get("frontend"),
        "split": payload.get("split"),
        "input_stream_sha256": input_sha256,
        "required_shards": required_shards,
        "completed_shards": len(entries),
        "checked_tracks": total_tracks,
        "checked_observations": total_observations,
        "unique_sample_keys": len(seen_keys),
        "unique_physical_tracks": len(seen_physical_ids),
        "invalid_box_count": invalid_boxes,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "shards": reports,
        "errors": errors[:100],
        "error_count": len(errors),
        "formal_cache_complete": complete_coverage,
        "test_content_opened": False,
        "external_processes_touched": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = audit(args.manifest, args.expected_source_sha256)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] in {"PASS_COMPLETE", "PASS_PARTIAL_COMPLETED_SHARDS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
