#!/usr/bin/env python3
"""Audit the public predicted-track v2 feature cache.

The predicted cache is much larger than the GT-track cache, so this audit is
streaming with respect to feature payloads: it keeps only the manifest key set
and aggregate counters in memory.  It validates every completed feature file
and never reads evaluator labels.
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


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST = OUTPUT_TARGET / "manifests/tao_val_predicted_tracks.jsonl"
FEATURE_ROOT = OUTPUT_TARGET / "features/pred_tracks/pred"


def _manifest_keys() -> tuple[set[str], str, int, int]:
    expected: set[str] = set()
    tracks = observations = 0
    with MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["sample_key"])
            if key in expected:
                raise ValueError(f"duplicate predicted sample_key: {key}")
            expected.add(key)
            tracks += 1
            observations += len(row["frame_ids"])
    return expected, sha256_file(MANIFEST), tracks, observations


def _audit_file(path: Path, expected: set[str], seen: set[str], prefix_counts: Counter, dimensions: Counter) -> dict | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        key = str(row["sample_key"])
        if key in seen:
            raise ValueError(f"duplicate feature sample_key: {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"feature key is absent from public manifest: {key}")
        forbidden = {"gt_category_id", "gt_split", "gt_category_name", "gt_match_id"}
        leaked = sorted(forbidden & set(row))
        if leaked:
            raise ValueError(f"predicted feature contains evaluator fields: {leaked}")
        frames = np.asarray(row["frame_embeddings"], dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != 768 or not np.isfinite(frames).all():
            raise ValueError(f"invalid frame feature matrix: {frames.shape}")
        lengths = [row["frame_ids"], row["image_paths"], row["boxes_xyxy"], row["quality"], row["frame_embeddings"]]
        if len({len(value) for value in lengths}) != 1 or frames.shape[0] != len(row["frame_ids"]):
            raise ValueError("lineage and feature lengths differ")
        dimensions[int(frames.shape[1])] += 1
        for prefix in PREFIXES:
            value = np.asarray(row["prefix_features"][str(prefix)], dtype=np.float32)
            if value.shape != (768,) or not np.isfinite(value).all():
                raise ValueError(f"invalid prefix feature p{prefix}: {value.shape}")
            prefix_counts[prefix] += 1
        full = np.asarray(row["full_feature"], dtype=np.float32)
        if full.shape != (768,) or not np.isfinite(full).all():
            raise ValueError(f"invalid full feature: {full.shape}")
        return None
    except Exception as exc:
        return {"path": str(path), "error": repr(exc)}


def main() -> int:
    out = ensure_output_layout()
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)
    expected, manifest_sha, expected_tracks, expected_observations = _manifest_keys()
    seen: set[str] = set()
    malformed: list[dict] = []
    prefix_counts: Counter = Counter()
    dimensions: Counter = Counter()
    file_count = 0
    failed_markers = 0
    failed_marker_examples: list[str] = []
    if FEATURE_ROOT.exists():
        for path in FEATURE_ROOT.glob("*.json"):
            file_count += 1
            error = _audit_file(path, expected, seen, prefix_counts, dimensions)
            if error is not None:
                malformed.append(error)
        for path in FEATURE_ROOT.glob("*.json.failed"):
            failed_markers += 1
            if len(failed_marker_examples) < 20:
                failed_marker_examples.append(str(path))

    missing = expected - seen
    unexpected = seen - expected
    ready = (
        file_count == expected_tracks
        and not missing
        and not unexpected
        and not malformed
        and failed_markers == 0
        and all(prefix_counts.get(prefix, 0) == expected_tracks for prefix in PREFIXES)
    )
    result = {
        "schema_version": "trackocd.v2.predicted_feature_audit.v1",
        "status": "READY" if ready else "PARTIAL",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": manifest_sha,
        "feature_root": str(FEATURE_ROOT.resolve()),
        "expected_tracks": expected_tracks,
        "expected_observations": expected_observations,
        "file_count": file_count,
        "sample_key_count": len(seen),
        "missing_expected_count": len(missing),
        "missing_expected_examples": sorted(missing)[:20],
        "unexpected_count": len(unexpected),
        "unexpected_examples": sorted(unexpected)[:20],
        "dimensions": dict(sorted(dimensions.items())),
        "prefix_counts": {str(key): int(value) for key, value in sorted(prefix_counts.items())},
        "malformed_count": len(malformed),
        "malformed": malformed,
        "failed_marker_count": failed_markers,
        "failed_marker_examples": failed_marker_examples,
        "model_input_contract": {
            "public_track_manifest_only": True,
            "gt_category_id_feature": False,
            "gt_split_feature": False,
            "gt_matching_used_to_build_features": False,
            "text_or_category_logits": False,
            "future_observations_used": False,
        },
        "test_semantic_accessed": False,
    }
    atomic_json(out / "audit/predicted_feature_audit.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
