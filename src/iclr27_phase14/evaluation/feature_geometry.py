"""Post-hoc Q1 geometry table for frozen representation candidates.

Private labels are loaded only for audit annotations after all feature values
are produced.  This module never emits online actions and never trains.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), 1e-12)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def summary(values):
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(len(a)), "mean": float(a.mean()),
            "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90))}


def geometry(records, vectors):
    ids = [r["sample_id"] for r in records]
    labels = np.asarray([int(r["category"]) for r in records])
    videos = np.asarray([int(r["video_id"]) for r in records])
    x = l2(np.stack([vectors[s] for s in ids]))
    same, different, pairs = [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = float(1.0 - np.dot(x[i], x[j]))
            if labels[i] == labels[j]:
                same.append(d)
                pairs.append({"left": ids[i], "right": ids[j],
                              "category": int(labels[i]),
                              "cross_video": bool(videos[i] != videos[j]),
                              "distance": d})
            else:
                different.append(d)
    nn, cross_nn = 0, 0
    for i in range(len(ids)):
        others = [j for j in range(len(ids)) if j != i]
        if others:
            j = min(others, key=lambda q: 1.0 - float(np.dot(x[i], x[q])))
            nn += int(labels[i] == labels[j])
        cross = [j for j in others if videos[j] != videos[i]]
        if cross:
            j = min(cross, key=lambda q: 1.0 - float(np.dot(x[i], x[q])))
            cross_nn += int(labels[i] == labels[j])
    # Leave-one-out category prototypes are reported only for categories with
    # at least two tracks.  Singleton novel categories cannot define a legal
    # post-hoc prototype, so they are excluded from this diagnostic rather
    # than counted as failures.
    by_cat = defaultdict(list)
    for i, c in enumerate(labels):
        by_cat[int(c)].append(i)
    proto_correct = 0
    proto_total = 0
    usable = {c for c, idxs in by_cat.items() if len(idxs) >= 2}
    for i, c in enumerate(labels):
        c = int(c)
        if c not in usable:
            continue
        centroids, cats = [], []
        for other, idxs in by_cat.items():
            pool = [j for j in idxs if j != i]
            if not pool:
                continue
            centroids.append(l2(x[pool].mean(axis=0)))
            cats.append(int(other))
        if not centroids:
            continue
        pred = cats[int(np.argmax(np.asarray(centroids) @ x[i]))]
        proto_correct += int(pred == c)
        proto_total += 1
    kmeans = None
    if len(ids) > 1:
        k = min(len(np.unique(labels)), len(ids))
        pred = KMeans(n_clusters=k, n_init=20, random_state=1027).fit(x).labels_
        kmeans = {"n_clusters": int(k),
                  "nmi": float(normalized_mutual_info_score(labels, pred)),
                  "ari": float(adjusted_rand_score(labels, pred))}
    return {
        "tracks": len(ids), "categories": int(len(np.unique(labels))),
        "same_category_pairs": len(same),
        "same_category_cross_video_pairs": int(sum(p["cross_video"] for p in pairs)),
        "same_category_distance": summary(same),
        "different_category_distance": summary(different),
        "distance_gap": (float(np.mean(different) - np.mean(same))
                          if same and different else None),
        "nearest_neighbor_accuracy": float(nn / max(len(ids), 1)),
        "cross_video_nearest_neighbor_accuracy": float(cross_nn / max(len(ids), 1)),
        "prototype_accuracy": float(proto_correct / max(proto_total, 1)),
        "prototype_n": int(proto_total),
        "kmeans": kmeans,
        "pairs": pairs,
    }


def load_records():
    from src.iclr27_phase4s.protocol import load_gt_tracks_dev
    stream, labels = load_gt_tracks_dev()
    records = []
    for row in stream:
        lab = labels[row["sample_id"]]
        if lab["protocol_role"] != "novel":
            continue
        records.append({"sample_id": row["sample_id"],
                        "video_id": int(row["video_id"]),
                        "track_id": int(row["track_id"]),
                        "category": int(lab["ground_truth_category_id"]),
                        "length": len(row["frame_ids"])})
    return records


def load_cache(root):
    out = {}
    for p in sorted((ROOT / root).glob("*.json")):
        d = json.loads(p.read_text())
        out[d["sample_id"]] = np.asarray(d["mean_embedding"], dtype=np.float32)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    records = load_records()
    aligned = set(json.loads((ROOT / "outputs/iclr27_phase11/eval/q1_correspondence_audit.json").read_text())
                  ["opportunity_audit"]["aligned_track_ids"])
    candidates = {
        "dino_v2": load_cache("data/caches/features/dinov2/gt_tracks_mean"),
        "clip_vit_b32": load_cache("data/caches/features/clip/gt_tracks_mean"),
        "dino_v3_cls": load_cache("data/caches/features/dinov3_vitb16_lvd1689m/gt_tracks/mean"),
        "dino_v3_pooled": load_cache("data/caches/features/dinov3_vitb16_lvd1689m_pooled/gt_tracks/mean"),
    }
    # TSE/B are already authoritative in Phase 11; retaining their exact
    # numbers avoids silently changing a frozen transform in this audit.
    prior = json.loads((ROOT / "outputs/iclr27_phase11/eval/q1_correspondence_audit.json").read_text())
    out = {"protocol": {"gt_used_posthoc_only": True, "n_all_novel": len(records),
                         "n_strict_aligned_novel": len(aligned)},
           "opportunity": prior["opportunity_audit"], "geometry": {}}
    aligned_records = [r for r in records if r["sample_id"] in aligned]
    for name, vec in candidates.items():
        ids = {r["sample_id"] for r in records}
        missing = sorted(ids - set(vec))
        if missing:
            out["geometry"][name] = {"status": "missing_cache", "missing": missing}
            continue
        out["geometry"][name] = {
            "all_gt_novel": geometry(records, vec),
            "strict_evaluable_aligned_novel": geometry(aligned_records, vec),
            "feature_dim": int(len(next(iter(vec.values())))),
        }
    def flatten_legacy(x):
        return {
            "tracks": int(x["n_tracks"]),
            "categories": int(x["n_categories"]),
            "same_category_pairs": int(x["same_category_pairs"]),
            "same_category_cross_video_pairs": int(x["same_category_cross_video_pairs"]),
            "distance_gap": x.get("inter_minus_intra"),
            "nearest_neighbor_accuracy": x.get("nearest_neighbor_accuracy"),
            "cross_video_nearest_neighbor_accuracy": x.get("cross_video_nearest_neighbor_accuracy"),
        }
    out["geometry"]["tse"] = {
        "all_gt_novel": flatten_legacy(prior["gt_cached_feature_geometry"]["all_gt_novel"]["tse"]),
        "strict_evaluable_aligned_novel": flatten_legacy(prior["gt_cached_feature_geometry"]["strict_evaluable_aligned_novel"]["tse"]),
        "feature_dim": 128,
    }
    out["geometry"]["phase8a_b"] = {
        "all_gt_novel": flatten_legacy(prior["gt_cached_feature_geometry"]["all_gt_novel"]["phase8a_b"]),
        "strict_evaluable_aligned_novel": flatten_legacy(prior["gt_cached_feature_geometry"]["strict_evaluable_aligned_novel"]["phase8a_b"]),
        "feature_dim": 128,
    }
    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: {s: {m: v.get(m) for m in ("tracks", "same_category_pairs", "same_category_cross_video_pairs", "distance_gap", "nearest_neighbor_accuracy", "cross_video_nearest_neighbor_accuracy")} for s, v in x.items() if isinstance(v, dict) and "status" not in v} for k, x in out["geometry"].items()}, indent=2))


if __name__ == "__main__":
    main()
