#!/usr/bin/env python3
"""Audit causal common-feature geometry without training or model selection."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.trackocd_v2.io import OUTPUT_TARGET, atomic_json, atomic_write_text, ensure_output_layout  # noqa: E402


PREFIXES = (1, 2, 4, 8, 16)
MANIFEST_ROOT = OUTPUT_TARGET / "manifests"
FEATURE_ROOT = OUTPUT_TARGET / "features/gt_tracks"


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def distribution_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.percentile(array, 50)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
    }


def auroc_positive_higher(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    scores = np.asarray(list(positive) + list(negative), dtype=np.float64)
    labels = np.asarray([1] * len(positive) + [0] * len(negative), dtype=np.int8)
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    n_pos, n_neg = float(len(positive)), float(len(negative))
    u = ranks[labels == 1].sum() - n_pos * (n_pos + 1.0) / 2.0
    return float(u / (n_pos * n_neg))


def separation(positive: Sequence[float], negative: Sequence[float]) -> Dict[str, Any]:
    if not positive or not negative:
        return {"positive_higher_auroc": None, "cohen_d": None, "overlap_coefficient": None}
    p, n = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    pooled = np.sqrt((float(p.var()) + float(n.var())) / 2.0)
    lo, hi = float(min(p.min(), n.min())), float(max(p.max(), n.max()))
    if hi <= lo:
        overlap = 1.0
    else:
        bins = np.linspace(lo, hi, 51)
        ph, _ = np.histogram(p, bins=bins, density=True)
        nh, _ = np.histogram(n, bins=bins, density=True)
        overlap = float(np.minimum(ph, nh).sum() * (bins[1] - bins[0]))
    return {
        "positive_higher_auroc": auroc_positive_higher(p, n),
        "cohen_d": float((p.mean() - n.mean()) / pooled) if pooled > 1e-12 else None,
        "overlap_coefficient": overlap,
    }


def sample_pairs(
    z: np.ndarray,
    frame_features: Sequence[np.ndarray],
    categories: Sequence[int],
    videos: Sequence[int],
    seed: int,
) -> Dict[str, List[float]]:
    """Build fixed-size deterministic identity/category pair populations."""

    rng = np.random.RandomState(seed)
    category_groups: Dict[int, List[int]] = defaultdict(list)
    for index, category in enumerate(categories):
        category_groups[int(category)].append(index)
    same_identity: List[float] = []
    for features in frame_features:
        if len(features) < 2:
            continue
        tries = min(32, len(features) * 2)
        for _ in range(tries):
            a, b = rng.choice(len(features), size=2, replace=False)
            same_identity.append(float(np.dot(features[a], features[b])))
    same_category: List[float] = []
    same_category_cross_video: List[float] = []
    for members in category_groups.values():
        if len(members) < 2:
            continue
        for _ in range(min(64, len(members) * 2)):
            a, b = rng.choice(members, size=2, replace=False)
            value = float(np.dot(z[a], z[b]))
            same_category.append(value)
            if videos[a] != videos[b]:
                same_category_cross_video.append(value)
    different_category: List[float] = []
    category_keys = sorted(category_groups)
    for _ in range(min(8192, len(categories) * 2)):
        a = int(rng.randint(0, len(categories)))
        other = category_keys[int(rng.randint(0, len(category_keys)))]
        if other == categories[a]:
            continue
        members = category_groups[other]
        b = members[int(rng.randint(0, len(members)))]
        different_category.append(float(np.dot(z[a], z[b])))
    return {
        "same_identity": same_identity,
        "same_category_diff_track": same_category,
        "same_category_diff_track_cross_video": same_category_cross_video,
        "different_category": different_category,
    }


def summarize_pairs(pairs: Dict[str, List[float]]) -> Dict[str, Any]:
    similarity = {name: distribution_stats(values) for name, values in pairs.items()}
    distance = {name: distribution_stats([1.0 - value for value in values]) for name, values in pairs.items()}
    return {
        "cosine_similarity": similarity,
        "cosine_distance": distance,
        "same_category_vs_different_category": separation(pairs["same_category_diff_track"], pairs["different_category"]),
        "same_category_cross_video_vs_different_category": separation(pairs["same_category_diff_track_cross_video"], pairs["different_category"]),
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
    if not FEATURE_ROOT.exists():
        waiting = {"schema_version": "trackocd.v2.geometry.v2", "status": "WAITING_FEATURES", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(result_path, waiting)
        atomic_json(out / "diagnostics/representation_geometry.json", waiting)
        return 2

    split_results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        manifest_path = MANIFEST_ROOT / ("tao_%s_gt_tracks.jsonl" % split)
        labels_path = MANIFEST_ROOT / ("private_tao_%s_gt_track_labels.jsonl" % split)
        if not manifest_path.exists() or not labels_path.exists():
            waiting = {"schema_version": "trackocd.v2.geometry.v2", "status": "WAITING_FEATURES", "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "missing_split": split}
            atomic_json(result_path, waiting)
            atomic_json(out / "diagnostics/representation_geometry.json", waiting)
            return 2
        manifest = load_jsonl(manifest_path)
        labels = {str(row["sample_key"]): row for row in load_jsonl(labels_path)}
        loaded: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        malformed: List[Dict[str, Any]] = []
        for row in manifest:
            key = str(row["sample_key"])
            path = FEATURE_ROOT / split / (key + ".json")
            if not path.exists():
                missing.append(key)
                continue
            try:
                feature = json.loads(path.read_text(encoding="utf-8"))
                prefixes = {str(prefix): np.asarray(feature["prefix_features"][str(prefix)], dtype=np.float32) for prefix in PREFIXES}
                frames = unit_rows(np.asarray(feature["frame_embeddings"], dtype=np.float32))
                if frames.ndim != 2 or frames.shape[1] != 768 or not np.isfinite(frames).all():
                    raise ValueError("invalid frame feature matrix")
                if any(value.shape != (768,) or not np.isfinite(value).all() for value in prefixes.values()):
                    raise ValueError("invalid prefix vector")
                loaded[key] = {"prefixes": prefixes, "frames": frames}
            except Exception as exc:
                malformed.append({"sample_key": key, "error": repr(exc)})
        if missing or malformed or len(loaded) != len(manifest):
            waiting = {
                "schema_version": "trackocd.v2.geometry.v2",
                "status": "WAITING_FEATURES",
                "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "split": split,
                "expected_tracks": len(manifest),
                "loaded_tracks": len(loaded),
                "missing": missing,
                "malformed": malformed,
            }
            atomic_json(result_path, waiting)
            atomic_json(out / "diagnostics/representation_geometry.json", waiting)
            return 2

        usable = [row for row in manifest if labels[str(row["sample_key"])]["gt_split"] in ("old", "new")]
        keys = [str(row["sample_key"]) for row in usable]
        categories = [int(labels[key]["gt_category_id"]) for key in keys]
        videos = [int(labels[key]["video_id"]) for key in keys]
        physical = [str(labels[key]["physical_track_id"]) for key in keys]
        by_prefix = {}
        for prefix in PREFIXES:
            z = unit_rows(np.stack([loaded[key]["prefixes"][str(prefix)] for key in keys]))
            pairs = sample_pairs(
                z,
                [loaded[key]["frames"] for key in keys],
                categories,
                videos,
                args.seed + prefix + (0 if split == "train" else 1000),
            )
            by_prefix[str(prefix)] = {
                "tracks": len(keys),
                "dimension": int(z.shape[1]),
                "norm_min": float(np.linalg.norm(z, axis=1).min()),
                "norm_max": float(np.linalg.norm(z, axis=1).max()),
                "pair_metrics": summarize_pairs(pairs),
                "nearest_same_or_cross_video": nearest_recall(z, categories, videos, cross_video=False),
                "nearest_cross_video": nearest_recall(z, categories, videos, cross_video=True),
            }
            for group, values in pairs.items():
                stats = distribution_stats(values)
                csv_rows.append({
                    "split": split,
                    "prefix": prefix,
                    "group": group,
                    "count": stats["count"],
                    "mean_cosine": stats["mean"],
                    "std_cosine": stats["std"],
                    "p10_cosine": stats["p10"],
                    "p25_cosine": stats["p25"],
                    "p50_cosine": stats["p50"],
                    "p75_cosine": stats["p75"],
                    "p90_cosine": stats["p90"],
                })
        split_results[split] = {
            "tracks": len(keys),
            "categories": len(set(categories)),
            "videos": len(set(videos)),
            "physical_tracks": len(set(physical)),
            "prefixes": by_prefix,
        }

    result = {
        "schema_version": "trackocd.v2.geometry.v2",
        "status": "COMPLETE",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluator_only_labels": True,
        "model_input_fields": ["visual_features", "quality", "frame_ids", "boxes_xyxy"],
        "forbidden_representation_fields": ["gt_category_id", "gt_split", "category_text", "category_logits"],
        "splits": split_results,
        "protocol": "diagnostic geometry only; no learned selector, threshold, controller, or Test access",
    }
    atomic_json(result_path, result)
    atomic_json(out / "diagnostics/representation_geometry.json", result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["split", "prefix", "group", "count", "mean_cosine", "std_cosine", "p10_cosine", "p25_cosine", "p50_cosine", "p75_cosine", "p90_cosine"])
    writer.writeheader()
    writer.writerows(csv_rows)
    atomic_write_text(out / "diagnostics/representation_geometry.csv", buffer.getvalue())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
