#!/usr/bin/env python3
"""Audit causal common-feature geometry without training or model selection."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, ensure_output_layout  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST = OUTPUT_TARGET / "manifests/tao_val_gt_tracks.jsonl"
LABELS = OUTPUT_TARGET / "manifests/private_tao_val_gt_track_labels.jsonl"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks/val"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def pair_summary(z: np.ndarray, labels: Sequence[int], videos: Sequence[int], seed: int) -> Dict[str, Any]:
    """Sample legal same/different-category pairs for a stable geometry audit."""

    rng = np.random.RandomState(seed)
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, category in enumerate(labels):
        groups[int(category)].append(index)
    same: List[float] = []
    same_cross_video: List[float] = []
    different: List[float] = []
    keys = sorted(groups)
    for category in keys:
        members = groups[category]
        if len(members) >= 2:
            for _ in range(min(32, len(members) * 2)):
                a, b = rng.choice(members, size=2, replace=False)
                value = float(np.dot(z[a], z[b]))
                same.append(value)
                if videos[a] != videos[b]:
                    same_cross_video.append(value)
    if len(keys) >= 2:
        for _ in range(min(4096, len(labels) * 2)):
            a = int(rng.randint(0, len(labels)))
            other = keys[int(rng.randint(0, len(keys)))]
            if other == labels[a]:
                continue
            b = groups[other][int(rng.randint(0, len(groups[other])))]
            different.append(float(np.dot(z[a], z[b])))

    def stats(values: Sequence[float]) -> Dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "median": None, "p10": None, "p90": None}
        array = np.asarray(values, dtype=np.float32)
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
        }

    return {
        "same_category": stats(same),
        "same_category_cross_video": stats(same_cross_video),
        "different_category": stats(different),
        "same_minus_different_mean": (float(np.mean(same)) - float(np.mean(different))) if same and different else None,
    }


def nearest_recall(z: np.ndarray, categories: Sequence[int], videos: Sequence[int], cross_video: bool) -> Dict[str, Any]:
    similarities = np.matmul(z, z.T)
    np.fill_diagonal(similarities, -np.inf)
    if cross_video:
        video_array = np.asarray(videos)
        similarities[video_array[:, None] == video_array[None, :]] = -np.inf
    nearest = np.argmax(similarities, axis=1)
    valid = np.isfinite(similarities[np.arange(len(z)), nearest])
    if not valid.any():
        return {"queries": 0, "category_recall_at_1": None, "valid_fraction": 0.0}
    hits = np.asarray(categories)[nearest[valid]] == np.asarray(categories)[valid]
    return {
        "queries": int(valid.sum()),
        "category_recall_at_1": float(hits.mean()),
        "valid_fraction": float(valid.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    out = ensure_output_layout()
    result_path = out / "audit/geometry_audit.json"
    if not MANIFEST.exists() or not LABELS.exists() or not FEATURE_ROOT.exists():
        atomic_json(result_path, {"schema_version": "trackocd.v2.geometry.v1", "status": "WAITING_FEATURES", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
        return 2
    manifest = load_jsonl(MANIFEST)
    label_rows = load_jsonl(LABELS)
    labels = {str(row["sample_key"]): row for row in label_rows}
    vectors: Dict[str, Dict[str, np.ndarray]] = {}
    missing = []
    malformed = []
    for row in manifest:
        key = str(row["sample_key"])
        path = FEATURE_ROOT / (key + ".json")
        if not path.exists():
            missing.append(key)
            continue
        try:
            feature = json.loads(path.read_text(encoding="utf-8"))
            prefixes = {str(prefix): np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32) for prefix in PREFIXES}
            if any(value.shape != (768,) or not np.isfinite(value).all() for value in prefixes.values()):
                raise ValueError("invalid prefix vector")
            vectors[key] = prefixes
        except Exception as exc:
            malformed.append({"sample_key": key, "error": repr(exc)})
    if missing or malformed or len(vectors) != len(manifest):
        atomic_json(result_path, {
            "schema_version": "trackocd.v2.geometry.v1",
            "status": "WAITING_FEATURES",
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "expected_tracks": len(manifest),
            "loaded_tracks": len(vectors),
            "missing": missing,
            "malformed": malformed,
        })
        return 2

    usable = [row for row in manifest if labels[str(row["sample_key"])]["gt_split"] != "distractor"]
    keys = [str(row["sample_key"]) for row in usable]
    categories = [int(labels[key]["gt_category_id"]) for key in keys]
    videos = [int(labels[key]["video_id"]) for key in keys]
    by_prefix = {}
    for prefix in PREFIXES:
        z = unit_rows(np.stack([vectors[key][str(prefix)] for key in keys]))
        by_prefix[str(prefix)] = {
            "tracks": len(keys),
            "dimension": int(z.shape[1]),
            "norm_min": float(np.linalg.norm(z, axis=1).min()),
            "norm_max": float(np.linalg.norm(z, axis=1).max()),
            "pair_geometry": pair_summary(z, categories, videos, args.seed + prefix),
            "nearest_same_or_cross_video": nearest_recall(z, categories, videos, cross_video=False),
            "nearest_cross_video": nearest_recall(z, categories, videos, cross_video=True),
        }
    result = {
        "schema_version": "trackocd.v2.geometry.v1",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluator_only_labels": True,
        "model_input_fields": ["visual_features", "quality", "frame_ids", "boxes_xyxy"],
        "forbidden_representation_fields": ["gt_category_id", "gt_split", "category_text", "category_logits"],
        "manifest": str(MANIFEST.resolve()),
        "label_sidecar": str(LABELS.resolve()),
        "tracks": len(keys),
        "categories": len(set(categories)),
        "videos": len(set(videos)),
        "prefixes": by_prefix,
        "protocol": "diagnostic geometry only; no learned selector, threshold, controller, or Test access",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
