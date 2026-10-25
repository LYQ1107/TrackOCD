#!/usr/bin/env python3
"""Recover a receipt after a completed pilot payload hit a terminal wrapper bug.

The recovery path is deliberately payload-only plus a fresh streaming source
contract check.  It never reruns the encoder, trains, or opens GT/Val/Test
assets.  A receipt is written only when the already-written payload and index
match the frozen configuration exactly.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.trackocd_cvpr import pilot_predicted_train_dino_features as pilot  # noqa: E402
from src.trackocd_v2.io import atomic_json, sha256_file  # noqa: E402


CONFIG = ROOT / "configs/trackocd_cvpr/predicted_train_policy_feature_pilot_v1.json"
RESULT = ROOT / "outputs/trackocd_cvpr/cvpr2/predicted_train_policy_feature_pilot_v1/result.json"


def check_index(path: Path, selected: list[dict[str, Any]], offsets: np.ndarray) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_keys = {
        "video_id", "physical_track_id", "sample_key", "source_sample_id",
        "source_split", "frame_ids", "offset", "count",
    }
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if set(row) != expected_keys:
                raise ValueError(f"unexpected index fields at line {line_number}")
            if row["source_split"] != "train_predicted" or int(row["video_id"]) != 72:
                raise ValueError(f"index boundary violation at line {line_number}")
            rows.append(row)
    if len(rows) != len(selected):
        raise ValueError(f"index track count mismatch: {len(rows)} != {len(selected)}")
    for index, (row, source_row) in enumerate(zip(rows, selected)):
        begin, end = int(offsets[index]), int(offsets[index + 1])
        if int(row["offset"]) != begin or int(row["count"]) != end - begin:
            raise ValueError(f"index offset mismatch at track {index}")
        if row["frame_ids"] != source_row["frame_ids"]:
            raise ValueError(f"index frame lineage mismatch at track {index}")


def main() -> int:
    if RESULT.exists():
        raise RuntimeError(f"refusing to overwrite existing receipt: {RESULT}")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    if cfg["fixed_selection"]["video_id"] != 72:
        raise ValueError("recovery is bound to fixed policy video 72")
    output = ROOT / cfg["output_directory"]
    payload = output / "video_0072_prefix_features.npz"
    index = output / "track_index.jsonl"
    progress = output / "progress.json"
    if not payload.is_file() or not index.is_file() or not progress.is_file():
        raise FileNotFoundError("completed pilot payload/index/progress is required")

    selected, source = pilot.scan_source(cfg)
    with np.load(payload, allow_pickle=False) as arrays:
        required = {
            "visual", "geometry_normalized_xyxy", "quality", "frame_ids",
            "track_offsets", "prefix_features",
        }
        if set(arrays.files) != required:
            raise ValueError(f"unexpected payload arrays: {arrays.files}")
        visual = arrays["visual"]
        geometry = arrays["geometry_normalized_xyxy"]
        quality = arrays["quality"]
        frame_ids = arrays["frame_ids"]
        offsets = arrays["track_offsets"]
        prefixes = arrays["prefix_features"]
        expected_obs = int(cfg["fixed_selection"]["expected_observations"])
        expected_tracks = int(cfg["fixed_selection"]["expected_tracks"])
        if visual.shape != (expected_obs, 768):
            raise ValueError(f"visual shape mismatch: {visual.shape}")
        if geometry.shape != (expected_obs, 4) or quality.shape != (expected_obs,):
            raise ValueError("geometry/quality shape mismatch")
        if frame_ids.shape != (expected_obs,) or offsets.shape != (expected_tracks + 1,):
            raise ValueError("frame/offset shape mismatch")
        if prefixes.shape != (expected_tracks, 5, 768):
            raise ValueError(f"prefix shape mismatch: {prefixes.shape}")
        for name, value in (("visual", visual), ("geometry", geometry), ("quality", quality),
                            ("frame_ids", frame_ids), ("offsets", offsets), ("prefixes", prefixes)):
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite {name}")
        if int(offsets[0]) != 0 or int(offsets[-1]) != expected_obs or np.any(np.diff(offsets) <= 0):
            raise ValueError("invalid track offsets")
        visual_norms = np.linalg.norm(visual.astype(np.float32), axis=1)
        prefix_norms = np.linalg.norm(prefixes.astype(np.float32), axis=2)
        if float(np.max(np.abs(visual_norms - 1.0))) > 3e-3:
            raise ValueError("visual features are not L2 normalized")
        if float(np.max(np.abs(prefix_norms - 1.0))) > 3e-3:
            raise ValueError("prefix features are not L2 normalized")
    check_index(index, selected, offsets)
    progress_data = json.loads(progress.read_text(encoding="utf-8"))
    if int(progress_data.get("forward_batches", -1)) != 793:
        raise ValueError("progress does not record all expected forward batches")

    payload_bytes = payload.stat().st_size
    index_bytes = index.stat().st_size
    result = {
        "schema_version": "trackocd.cvpr.predicted-train-policy-feature-pilot-result.v1",
        "status": "PASS_FIXED_POLICY_VIDEO_DINO_FEATURE_PILOT_NO_TRAINING_RECOVERED_RECEIPT",
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256_file(CONFIG),
        "script": str((ROOT / "scripts/trackocd_cvpr/pilot_predicted_train_dino_features.py").relative_to(ROOT)),
        "script_sha256": sha256_file(ROOT / "scripts/trackocd_cvpr/pilot_predicted_train_dino_features.py"),
        "recovery_script": str(Path(__file__).relative_to(ROOT)),
        "recovery_script_sha256": sha256_file(Path(__file__).resolve()),
        "source": source,
        "fixed_selection": cfg["fixed_selection"],
        "encoder": {
            **cfg["encoder"],
            "repository_commit_verified": True,
            "checkpoint_hash_verified": True,
            "frozen": True,
            "training_mode": False,
        },
        "features": {
            "tracks": len(selected),
            "observations": int(cfg["fixed_selection"]["expected_observations"]),
            "prefixes": [1, 2, 4, 8, 16],
            "arrays": ["visual", "geometry_normalized_xyxy", "quality", "frame_ids", "track_offsets", "prefix_features"],
            "payload_local_path": str(payload.relative_to(ROOT)),
            "payload_bytes": payload_bytes,
            "payload_sha256": sha256_file(payload),
            "index_local_path": str(index.relative_to(ROOT)),
            "index_bytes": index_bytes,
            "index_sha256": sha256_file(index),
            "payload_uploaded_to_github": False,
        },
        "resources": {
            "device_visible": "2",
            "batch_size": 32,
            "forward_batches": int(progress_data["forward_batches"]),
            "wall_seconds": float(progress_data["elapsed_seconds"]),
            "peak_rss_bytes": None,
            "peak_gpu_reserved_bytes": None,
            "peak_gpu_allocated_bytes": None,
            "disk_free_bytes_after": int(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize),
            "training_gpu_hours": 0.0,
        },
        "boundaries": cfg["boundaries"],
        "recovery": {
            "payload_forward_completed": True,
            "terminal_failure": "relative_config_path_in_receipt_serialization",
            "reran_encoder": False,
            "revalidated_source_and_payload": True,
        },
        "interpretation": "Bounded policy-input interface pilot only; no formal v2 cache, no MOT/OCD metric, no model-selection result, and no scientific claim.",
    }
    atomic_json(RESULT, result)
    print(json.dumps({"status": result["status"], "tracks": len(selected), "observations": result["features"]["observations"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
