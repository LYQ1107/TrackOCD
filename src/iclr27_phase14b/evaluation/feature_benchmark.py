"""Compute preregistered head-agnostic DEV+ representation metrics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase14b"
MANIFEST = OUT / "manifests/devplus_tracks.jsonl"
PREFIXES = (1, 2, 4, 8, 16)
BOOT = 400
SEED = 20260823


def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def ap_for_query(order, positive):
    hits = np.asarray([bool(x in positive) for x in order], dtype=np.float64)
    total = float(hits.sum())
    if total == 0:
        return None
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / total)


def grouped_ci(values_by_group, seed=SEED, repeats=BOOT):
    groups = sorted(values_by_group)
    if not groups:
        return {"groups": 0, "mean": None, "low": None, "high": None}
    group_values = [np.asarray(values_by_group[g], dtype=np.float64) for g in groups]
    point = float(np.concatenate(group_values).mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        picked = rng.integers(0, len(groups), size=len(groups))
        samples[i] = np.concatenate([group_values[j] for j in picked]).mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"groups": len(groups), "mean": point, "low": float(low), "high": float(high)}


def track_metrics(vectors, labels, videos):
    n = len(labels)
    sim = vectors @ vectors.T
    same_pairs, diff_pairs, cross_same_pairs = [], [], []
    same_dist_by_cat, diff_dist_by_cat = {}, {}
    for i in range(n):
        for j in range(i):
            distance = float(1.0 - sim[i, j])
            if labels[i] == labels[j]:
                same_pairs.append(distance)
                same_dist_by_cat.setdefault(int(labels[i]), []).append(distance)
                if videos[i] != videos[j]:
                    cross_same_pairs.append((i, j))
            else:
                diff_pairs.append(distance)
                diff_dist_by_cat.setdefault(int(labels[i]), []).append(distance)
    same_queries = {int(c): [] for c in sorted(set(labels))}
    cross_queries = {int(c): [] for c in sorted(set(labels))}
    same_video_queries = {int(v): [] for v in sorted(set(videos))}
    cross_video_map_values = {int(c): [] for c in sorted(set(labels))}
    proto_values = {int(c): [] for c in sorted(set(labels))}
    r1, r5, cross_r1, cross_r5 = [], [], [], []
    r1_by_cat, r5_by_cat, cr1_by_cat, cr5_by_cat = {}, {}, {}, {}
    for i in range(n):
        candidates = np.asarray([j for j in range(n) if j != i], dtype=np.int64)
        order = candidates[np.argsort(-sim[i, candidates], kind="stable")]
        positives = {j for j in order if labels[j] == labels[i]}
        if positives:
            hit1 = float(order[0] in positives)
            hit5 = float(any(j in positives for j in order[:5]))
            r1.append(hit1); r5.append(hit5)
            same_queries[int(labels[i])].append(hit1)
            r1_by_cat.setdefault(int(labels[i]), []).append(hit1)
            r5_by_cat.setdefault(int(labels[i]), []).append(hit5)
        cross_candidates = np.asarray([j for j in candidates if videos[j] != videos[i]], dtype=np.int64)
        cross_order = cross_candidates[np.argsort(-sim[i, cross_candidates], kind="stable")] if len(cross_candidates) else np.asarray([], dtype=np.int64)
        cross_positives = {j for j in cross_order if labels[j] == labels[i]}
        if cross_positives:
            hit1 = float(cross_order[0] in cross_positives)
            hit5 = float(any(j in cross_positives for j in cross_order[:5]))
            cross_r1.append(hit1); cross_r5.append(hit5)
            cross_queries[int(labels[i])].append(hit1)
            same_video_queries[int(videos[i])].append(hit1)
            cr1_by_cat.setdefault(int(labels[i]), []).append(hit1)
            cr5_by_cat.setdefault(int(labels[i]), []).append(hit5)
            ap = ap_for_query(cross_order.tolist(), cross_positives)
            if ap is not None:
                cross_video_map_values.setdefault(int(labels[i]), []).append(ap)
        category_indices = np.where(labels == labels[i])[0]
        category_indices = category_indices[category_indices != i]
        if len(category_indices):
            centroid = norm(vectors[category_indices].mean(axis=0, keepdims=True))[0]
            predictions = [int(c) for c in sorted(set(labels))]
            centroids = {}
            for c in predictions:
                idx = np.where((labels == c) & (np.arange(n) != i))[0]
                if len(idx):
                    centroids[c] = norm(vectors[idx].mean(axis=0, keepdims=True))[0]
            if len(centroids) >= 2:
                pred = max(centroids, key=lambda c: float(vectors[i] @ centroids[c]))
                proto_values[int(labels[i])].append(float(pred == int(labels[i])))
    diff_mean = float(np.mean(diff_pairs)) if diff_pairs else None
    same_mean = float(np.mean(same_pairs)) if same_pairs else None
    return {
        "n_tracks": n,
        "n_same_category_pairs": len(same_pairs),
        "n_different_category_pairs": len(diff_pairs),
        "n_cross_video_same_category_pairs": len(cross_same_pairs),
        "same_category_distance_mean": same_mean,
        "different_category_distance_mean": diff_mean,
        "distance_gap_different_minus_same": (diff_mean - same_mean) if same_mean is not None and diff_mean is not None else None,
        "same_category_recall_at_1": {"value": float(np.mean(r1)) if r1 else None, "queries": len(r1), "categories": len(r1_by_cat)},
        "same_category_recall_at_5": {"value": float(np.mean(r5)) if r5 else None, "queries": len(r5), "categories": len(r5_by_cat)},
        "cross_video_recall_at_1": {"value": float(np.mean(cross_r1)) if cross_r1 else None, "queries": len(cross_r1), "categories": len(cr1_by_cat)},
        "cross_video_recall_at_5": {"value": float(np.mean(cross_r5)) if cross_r5 else None, "queries": len(cross_r5), "categories": len(cr5_by_cat)},
        "cross_video_map": {"value": float(np.mean([x for xs in cross_video_map_values.values() for x in xs])) if any(cross_video_map_values.values()) else None, "queries": sum(len(x) for x in cross_video_map_values.values()), "categories": sum(bool(x) for x in cross_video_map_values.values())},
        "prototype_accuracy": {"value": float(np.mean([x for xs in proto_values.values() for x in xs])) if any(proto_values.values()) else None, "queries": sum(len(x) for x in proto_values.values()), "categories": sum(bool(x) for x in proto_values.values())},
        "category_macro": {
            "same_r1": grouped_ci({c: v for c, v in r1_by_cat.items()}),
            "same_r5": grouped_ci({c: v for c, v in r5_by_cat.items()}),
            "cross_video_r1": grouped_ci({c: v for c, v in cr1_by_cat.items()}),
            "cross_video_r5": grouped_ci({c: v for c, v in cr5_by_cat.items()}),
            "cross_video_map": grouped_ci({c: v for c, v in cross_video_map_values.items() if v}),
            "prototype": grouped_ci({c: v for c, v in proto_values.items() if v}),
        },
        "video_grouped_cross_video_r1": grouped_ci({v: x for v, x in same_video_queries.items() if x}),
    }


def main():
    rows = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("split") == "devplus" and r.get("role") == "devplus_novel"]
    rows.sort(key=lambda r: int(r["chronological_position"]))
    labels = np.asarray([int(r["category_id"]) for r in rows], dtype=np.int64)
    videos = np.asarray([int(r["video_id"]) for r in rows], dtype=np.int64)
    candidates = {}
    for name in ("dinov2", "clip", "dinov3"):
        path = OUT / "features" / f"devplus_{name}_gtbox.npz"
        z = np.load(path, allow_pickle=False)
        keys = [str(x) for x in z["sample_keys"]]
        expected = [r["sample_id"] for r in rows]
        if keys != expected:
            raise RuntimeError(f"{name}: cache keys do not match manifest")
        prefix_metrics = {}
        for prefix in PREFIXES:
            mask = z["frame_mask"][:, :prefix].astype(np.float32)
            frame = z["frame_features"][:, :prefix].astype(np.float32)
            vec = norm((frame * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1.0))
            prefix_metrics[str(prefix)] = track_metrics(vec, labels, videos)
        candidates[name] = {
            "status": "measured_gt_box_diagnostic",
            "feature_cache": str(path),
            "prefixes": prefix_metrics,
            "q1_used": False,
            "private_gt_used_for_features": False,
            "future_frames_used": False,
            "physical_id_used_as_feature": False,
        }
    result = {
        "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
        "selection_locked_before_results": True,
        "q1_used": False,
        "opportunity_source": str(OUT / "eval/opportunity_audit.json"),
        "bootstrap": {"repeats": BOOT, "seed": SEED, "groups": ["category", "query_video"]},
        "candidates": candidates,
        "not_executed_yet": {
            "object_centric_entity": ["TrackVerse", "VESSA", "SRL"],
            "streaming_video_semantic": ["InternVideo2.5", "StreamFormer"],
            "tracking_aware_correspondence": ["MoSiC", "TRACT", "COVTrack", "OVTR", "SimOWT"],
        },
    }
    path = OUT / "eval/foundation_feature_benchmark.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(path)
    print(json.dumps({name: {p: candidates[name]["prefixes"][p]["cross_video_recall_at_1"] for p in map(str, PREFIXES)} for name in candidates}, indent=2))


if __name__ == "__main__":
    main()
