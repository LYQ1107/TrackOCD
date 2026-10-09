#!/usr/bin/env python3
"""Audit existing generic DINOv2 track features for v2 reuse.

The cache is historical but category-agnostic.  This script does not copy it;
it records exact coverage and exposes a missing-prefix gate before geometry
experiments are allowed to claim completion.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout, sha256_file  # noqa: E402


FEATURE_ROOT = ROOT / "data/caches/features/dinov2"
V2_FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"
MANIFEST_ROOT = ROOT / "outputs/trackocd_v2/manifests"


def load_ids(path: Path) -> set[str]:
    return {json.loads(line)["sample_key"] for line in path.read_text().splitlines() if line.strip()}


def audit_directory(directory: Path, expected: set[str] | None = None) -> dict:
    paths = sorted(directory.glob("*.json"))
    keys = set()
    dimensions = Counter()
    frame_counts = Counter()
    missing_mean = []
    malformed = []
    for path in paths:
        try:
            row = json.loads(path.read_text())
            key = str(row["sample_id"])
            keys.add(key)
            mean = np.asarray(row["mean_embedding"], dtype=np.float32)
            frames = np.asarray(row["frame_embeddings"], dtype=np.float32)
            dimensions[int(mean.shape[0])] += 1
            frame_counts[int(frames.shape[0])] += 1
            if frames.ndim != 2 or frames.shape[1] != mean.shape[0] or not np.isfinite(frames).all():
                malformed.append(str(path))
            if mean.shape[0] == 0:
                missing_mean.append(str(path))
        except Exception:
            malformed.append(str(path))
    expected = expected or set()
    return {
        "path": str(directory.resolve()),
        "file_count": len(paths),
        "sample_key_count": len(keys),
        "expected_key_count": len(expected),
        "missing_expected_keys": sorted(expected - keys),
        "unexpected_keys": sorted(keys - expected),
        "dimensions": dict(sorted(dimensions.items())),
        "frame_count_distribution": dict(sorted(frame_counts.items())),
        "malformed": malformed,
        "missing_mean": missing_mean,
        "file_list_sha256": __import__("hashlib").sha256("\n".join(f"{p.name}:{p.stat().st_size}" for p in paths).encode()).hexdigest(),
    }


def audit_v2_directory(directory: Path, expected: set[str]) -> dict:
    """Audit the new per-track cache, including all causal aggregates."""

    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    keys = set()
    malformed = []
    prefix_counts = Counter()
    dimensions = Counter()
    for path in paths:
        try:
            row = json.loads(path.read_text())
            key = str(row["sample_key"])
            keys.add(key)
            frame = np.asarray(row["frame_embeddings"], dtype=np.float32)
            if frame.ndim != 2:
                raise ValueError("frame_embeddings is not a matrix")
            dimensions[int(frame.shape[1])] += 1
            lengths = [row["frame_ids"], row["image_paths"], row["boxes_xyxy"], row["quality"], row["frame_embeddings"]]
            if len({len(value) for value in lengths}) != 1 or frame.shape[0] != len(row["frame_ids"]):
                raise ValueError("lineage and feature lengths differ")
            if frame.shape[1] != 768 or not np.isfinite(frame).all():
                raise ValueError("invalid frame feature matrix")
            prefixes = row["prefix_features"]
            for prefix in (1, 2, 4, 8, 16):
                value = np.asarray(prefixes[str(prefix)], dtype=np.float32)
                if value.shape != (768,) or not np.isfinite(value).all():
                    raise ValueError("invalid prefix feature")
                prefix_counts[prefix] += 1
            full = np.asarray(row["full_feature"], dtype=np.float32)
            if full.shape != (768,) or not np.isfinite(full).all():
                raise ValueError("invalid full feature")
        except Exception as exc:
            malformed.append({"path": str(path), "error": repr(exc)})
    return {
        "path": str(directory.resolve()),
        "file_count": len(paths),
        "sample_key_count": len(keys),
        "expected_key_count": len(expected),
        "missing_expected_keys": sorted(expected - keys),
        "unexpected_keys": sorted(keys - expected),
        "dimensions": dict(sorted(dimensions.items())),
        "prefix_counts": {str(key): int(value) for key, value in sorted(prefix_counts.items())},
        "malformed": malformed,
    }


def main() -> None:
    out = ensure_output_layout()
    val_manifest = MANIFEST_ROOT / "tao_val_gt_tracks.jsonl"
    train_manifest = MANIFEST_ROOT / "tao_train_gt_tracks.jsonl"
    val_expected = load_ids(val_manifest)
    train_expected = load_ids(train_manifest)
    train_known_manifest = set()
    labels = MANIFEST_ROOT / "private_tao_train_gt_track_labels.jsonl"
    if labels.exists():
        for line in labels.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row["gt_split"] == "old":
                    train_known_manifest.add(row["sample_key"])
    v2_train = audit_v2_directory(V2_FEATURE_ROOT / "train", train_expected)
    v2_val = audit_v2_directory(V2_FEATURE_ROOT / "val", val_expected)
    v2_ready = all(
        not audit["missing_expected_keys"]
        and not audit["unexpected_keys"]
        and not audit["malformed"]
        and audit["sample_key_count"] == audit["expected_key_count"]
        and all(audit["prefix_counts"].get(str(prefix), 0) == audit["expected_key_count"] for prefix in (1, 2, 4, 8, 16))
        for audit in (v2_train, v2_val)
    )
    result = {
        "schema_version": "trackocd.v2.common_feature_audit.v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "feature": {
            "encoder": "DINOv2 ViT-B/14",
            "modality": "visual-only crop embedding",
            "text_or_category_logits": False,
            "physical_id_feature": False,
            "source_code": str((ROOT / "src/features/extract.py").resolve()),
            "source_code_sha256": sha256_file(ROOT / "src/features/extract.py"),
            "cache_root": str(FEATURE_ROOT.resolve()),
        },
        "v2_cache": {
            "root": str(V2_FEATURE_ROOT.resolve()),
            "builder": str((ROOT / "scripts/trackocd_v2/build_common_features.py").resolve()),
            "builder_sha256": sha256_file(ROOT / "scripts/trackocd_v2/build_common_features.py"),
            "train": v2_train,
            "val": v2_val,
        },
        "gt_val": audit_directory(FEATURE_ROOT / "gt_tracks_mean", val_expected),
        "train_all": audit_directory(FEATURE_ROOT / "full_tao_train", train_expected),
        "train_known": audit_directory(FEATURE_ROOT / "train_known_mean", train_known_manifest),
        "prefix_contract": {
            "required": [1, 2, 4, 8, 16],
            "available_from_existing_cache": [1, 2, 4, 8],
            "status": "READY" if v2_ready else "PARTIAL_NEEDS_P16",
            "geometry_authorized": bool(v2_ready),
        },
        "historical_cache_reused_read_only": True,
    }
    atomic_json(out / "audit/common_feature_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["prefix_contract"]["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
