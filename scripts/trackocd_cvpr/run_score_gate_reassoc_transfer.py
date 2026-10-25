#!/usr/bin/env python3
"""Run one fixed category-free score-gate/reassociation transfer audit.

The transform reads existing anonymous ``train_predicted`` physical streams,
keeps two predeclared Train role subsets, applies the already registered
quality cut at 0.005, and then applies the fixed one-frame causal physical
reassociation.  It never reads GT, labels, vocabulary, pixels, features, or
Val/Test data.  The native subset is retained only as a local paired baseline.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr.causal_physical_reassociate import (
    FORBIDDEN_FIELDS,
    PHYSICAL_FIELDS,
    reassociate_video,
)


CONFIG = ROOT / "configs/trackocd_cvpr/covtrack_score_gate_reassoc_transfer_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{__import__('os').getpid()}")
    if temporary.exists() or path.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {path}")
    digest = hashlib.sha256()
    with temporary.open("xb", encoding="utf-8") as handle:
        for row in rows:
            encoded = (json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n").encode()
            handle.write(encoded.decode())
            digest.update(encoded)
        handle.flush()
    temporary.replace(path)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest(), "rows": len(rows)}


def _validate_row(row: dict[str, Any], path: Path, line_number: int) -> None:
    if set(row) != PHYSICAL_FIELDS:
        raise ValueError(f"physical stream schema mismatch at {path}:{line_number}")
    if FORBIDDEN_FIELDS & set(row) or row.get("source_split") != "train_predicted":
        raise ValueError(f"forbidden/non-Train row at {path}:{line_number}")
    frame_ids = row["frame_ids"]
    boxes = row["boxes_xyxy"]
    image_paths = row["image_paths"]
    quality = row["quality"]
    if (
        not isinstance(frame_ids, list)
        or not isinstance(boxes, list)
        or not isinstance(image_paths, list)
        or not isinstance(quality, list)
        or not frame_ids
        or len(frame_ids) != len(boxes)
        or len(frame_ids) != len(image_paths)
        or len(frame_ids) != len(quality)
        or len(set(frame_ids)) != len(frame_ids)
    ):
        raise ValueError(f"invalid lineage at {path}:{line_number}")
    for value in quality:
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid quality at {path}:{line_number}")


def _filtered_row(row: dict[str, Any], threshold: float) -> dict[str, Any] | None:
    kept = [
        index
        for index, value in enumerate(row["quality"])
        if float(value) >= threshold
    ]
    if not kept:
        return None
    result = dict(row)
    for field in ("frame_ids", "boxes_xyxy", "image_paths", "quality"):
        result[field] = [row[field][index] for index in kept]
    return result


def _load_fixed_rows(
    source_root: Path,
    role_ids: dict[str, set[int]],
    threshold: float,
) -> tuple[dict[str, dict[int, list[dict[str, Any]]]], dict[str, dict[int, list[dict[str, Any]]]], list[dict[str, Any]]]:
    paths = sorted(source_root.glob("*/physical_tracks.jsonl"))
    if len(paths) != 13:
        raise ValueError(f"expected exactly 13 completed physical streams, found {len(paths)}")
    all_ids = set().union(*role_ids.values())
    if sum(len(ids) for ids in role_ids.values()) != len(all_ids):
        raise ValueError("role video subsets overlap")
    native: dict[str, dict[int, list[dict[str, Any]]]] = {
        role: defaultdict(list) for role in role_ids
    }
    filtered: dict[str, dict[int, list[dict[str, Any]]]] = {
        role: defaultdict(list) for role in role_ids
    }
    seen_tracks: set[tuple[int, str]] = set()
    source_records: list[dict[str, Any]] = []
    for path in paths:
        selected: set[int] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object row at {path}:{line_number}")
                _validate_row(row, path, line_number)
                video_id = int(row["video_id"])
                roles = [role for role, ids in role_ids.items() if video_id in ids]
                if not roles:
                    continue
                if len(roles) != 1:
                    raise ValueError(f"ambiguous role for video {video_id}")
                role = roles[0]
                track_key = (video_id, str(row["physical_track_id"]))
                if track_key in seen_tracks:
                    raise ValueError(f"duplicate physical track {track_key}")
                seen_tracks.add(track_key)
                selected.add(video_id)
                native[role][video_id].append(row)
                candidate = _filtered_row(row, threshold)
                if candidate is not None:
                    filtered[role][video_id].append(candidate)
        source_records.append({
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "selected_videos": sorted(selected),
        })
    for role, ids in role_ids.items():
        if set(native[role]) != ids:
            raise ValueError(f"native stream videos for {role} do not match frozen IDs")
        if not set(filtered[role]).issubset(ids):
            raise ValueError(f"filtered stream escaped frozen IDs for {role}")
    return native, filtered, source_records


def _prepare_output_rows(rows_by_video: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_id in sorted(rows_by_video):
        rows.extend(rows_by_video[video_id])
    for stream_order, row in enumerate(rows):
        row["stream_order"] = stream_order
    return rows


def main() -> int:
    started = time.monotonic()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg.get("status") != "FROZEN_CATEGORY_FREE_SCORE_GATE_REASSOCIATION_TRANSFER_NO_TRAINING":
        raise ValueError("unexpected protocol status")
    source_root = ROOT / cfg["source"]["stream_root"]
    roles = {
        role: {int(video_id) for video_id in spec["video_ids"]}
        for role, spec in cfg["fixed_roles"].items()
    }
    threshold = float(cfg["fixed_mechanism"]["score_threshold"])
    native, filtered, source_records = _load_fixed_rows(source_root, roles, threshold)
    output_root = ROOT / "outputs/trackocd_cvpr/cvpr2/covtrack_score_gate_reassoc_transfer_v1"
    role_results: dict[str, Any] = {}
    for role in roles:
        native_rows = _prepare_output_rows({video: list(rows) for video, rows in native[role].items()})
        candidate_by_video: dict[int, list[dict[str, Any]]] = {}
        summaries: dict[str, Any] = {}
        for video_id in sorted(roles[role]):
            rows, summary = reassociate_video(
                filtered[role].get(video_id, []),
                iou_threshold=float(cfg["fixed_mechanism"]["reassociation_iou_threshold"]),
                max_gap_frames=int(cfg["fixed_mechanism"]["reassociation_max_gap_frames"]),
            )
            candidate_by_video[video_id] = rows
            summaries[str(video_id)] = summary
        candidate_rows = _prepare_output_rows(candidate_by_video)
        native_output = ROOT / cfg["source"]["native_subset_outputs"][role]
        candidate_output = ROOT / cfg["source"]["candidate_outputs"][role]
        native_meta = _write_rows(native_output, native_rows)
        candidate_meta = _write_rows(candidate_output, candidate_rows)
        role_results[role] = {
            "video_ids": sorted(roles[role]),
            "native_subset": native_meta,
            "candidate": candidate_meta,
            "native_tracks": len(native_rows),
            "native_observations": sum(len(row["frame_ids"]) for row in native_rows),
            "score_retained_tracks": sum(len(rows) for rows in filtered[role].values()),
            "score_retained_observations": sum(
                len(row["frame_ids"]) for rows in filtered[role].values() for row in rows
            ),
            "reassociation_by_video": summaries,
            "candidate_observations": sum(len(row["frame_ids"]) for row in candidate_rows),
        }
    result = {
        "schema_version": "trackocd.cvpr.covtrack-score-gate-reassociation-transfer-result.v1",
        "status": "COMPLETE_CATEGORY_FREE_SCORE_GATE_REASSOCIATION_TRANSFER",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "source": {
            "root": str(source_root.resolve()),
            "reconciliation": str((ROOT / cfg["source"]["reconciliation"]).resolve()),
            "reconciliation_sha256": cfg["source"]["reconciliation_sha256"],
            "stream_count": len(source_records),
            "streams": source_records,
        },
        "roles": role_results,
        "mechanism": cfg["fixed_mechanism"],
        "boundaries": cfg["boundaries"],
        "resources": {
            "device": "CPU",
            "wall_seconds": time.monotonic() - started,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        },
        "interpretation": "Category-free transfer evidence only; no official OCD/MOT metric, frontend qualification, model selection, predicted training, or Goal A/B/C acceptance.",
    }
    receipt = output_root / "run.json"
    if receipt.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {receipt}")
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "roles": role_results, "receipt": str(receipt)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
