"""DEV+-only learnability controls.

The controls deliberately stay outside the primary TrackOCD replay because the
proposal-aligned stream is not available yet.  They use one vector per GT
track, in chronological order, and report whether a representation can express
cross-instance correspondence.  The oracle is illegal and label-using; the
instance-only baseline uses only within-track feature changes; the supervised
diagnostic fits one fixed PCA+LDA metric on public representation-training
tracks and never reads DEV+ labels until evaluation.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.iclr27_phase14b.evaluation.feature_benchmark import norm, track_metrics

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = ROOT / "outputs/iclr27_phase14b"
SEED = 20260823


def load_manifest():
    rows = [json.loads(x) for x in (OUT / "manifests/devplus_tracks.jsonl").read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r["split"] == "devplus" and r["role"] == "devplus_novel"]
    rows.sort(key=lambda r: int(r["chronological_position"]))
    return rows


def load_features(rows):
    z = np.load(OUT / "features/devplus_dinov2_gtbox.npz", allow_pickle=False)
    keys = [str(x) for x in z["sample_keys"]]
    expected = [r["sample_id"] for r in rows]
    if keys != expected:
        raise RuntimeError("DEV+ feature keys do not match the locked manifest")
    frame = z["frame_features"].astype(np.float32)
    mask = z["frame_mask"].astype(np.float32)
    pooled = norm((frame * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1.0))
    # This removes static appearance and keeps only temporal change.  It is a
    # continuity-only control, not an object/category representation.
    delta = np.diff(frame, axis=1)
    dmask = mask[:, 1:] * mask[:, :-1]
    continuity = norm((delta * dmask[..., None]).sum(1) / np.maximum(dmask.sum(1, keepdims=True), 1.0))
    return pooled, continuity


def action_oracle(rows):
    seen_any = set()
    seen_videos = defaultdict(set)
    first = reuse = cross = 0
    actions = []
    for row in rows:
        c, v = int(row["category_id"]), int(row["video_id"])
        if c in seen_any:
            actions.append("EXISTING_NOVEL")
            reuse += 1
            if any(prev != v for prev in seen_videos[c]):
                cross += 1
        else:
            actions.append("NEW_NOVEL")
            first += 1
        seen_any.add(c)
        seen_videos[c].add(v)
    return {
        "actions": {"new_novel": first, "existing_novel": reuse},
        "devplus_novel_tracks": len(rows),
        "novel_birth_accuracy": 1.0 if first else None,
        "novel_reuse_accuracy": 1.0 if reuse else None,
        "cross_video_reuse_accuracy": 1.0 if cross else None,
        "novel_birth_denominator": first,
        "novel_reuse_denominator": reuse,
        "cross_video_reuse_denominator": cross,
    }


def known_support_oracle():
    """Report the known-route control on public TRAIN support only."""
    split = json.loads((OUT / "manifests/devplus_split.json").read_text())
    train_videos = set(int(x) for x in split["representation_train_videos"])
    z = np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", allow_pickle=False)
    keep = np.asarray([int(v) in train_videos for v in z["video_ids"]], dtype=bool)
    keep &= z["is_known"].astype(bool)
    return {
        "status": "measured_public_train_support_only",
        "known_occurrence_accuracy": 1.0 if int(keep.sum()) else None,
        "known_occurrence_denominator": int(keep.sum()),
        "known_categories": int(len(np.unique(z["labels"][keep]))) if int(keep.sum()) else 0,
        "category_label_used": True,
        "q1_used": False,
        "note": "Known routing is an oracle support control on public representation-training tracks, not a DEV+ primary TrackOCD evaluator result.",
    }


def retrieval_action_metrics(vectors, rows):
    """Causal nearest-past diagnostic; no threshold is selected on DEV+."""
    labels = np.asarray([int(r["category_id"]) for r in rows], dtype=np.int64)
    videos = np.asarray([int(r["video_id"]) for r in rows], dtype=np.int64)
    correct_reuse = cross_queries = false_reuse = 0
    predicted_existing = 0
    for i in range(len(rows)):
        past = np.arange(i, dtype=np.int64)
        if len(past) == 0:
            continue
        order = past[np.argsort(-(vectors[past] @ vectors[i]), kind="stable")]
        j = int(order[0])
        same_prior = bool(np.any(labels[past] == labels[i]))
        cross_prior = bool(np.any((labels[past] == labels[i]) & (videos[past] != videos[i])))
        if same_prior:
            predicted_existing += 1
            if labels[j] == labels[i]:
                correct_reuse += 1
            else:
                false_reuse += 1
        if cross_prior:
            cross_queries += 1
            if labels[j] == labels[i] and videos[j] != videos[i]:
                # The exact prior nearest-neighbour action is a diagnostic
                # proxy for CT-Reuse, not the frozen TrackOCD evaluator.
                pass
    return {
        "causal_nearest_past_correct_reuse": correct_reuse / max(predicted_existing, 1),
        "causal_nearest_past_reuse_queries": predicted_existing,
        "causal_nearest_past_false_reuse": false_reuse,
        "cross_video_queries_with_prior": cross_queries,
    }


def fit_supervised_projection():
    """Fit the one preregistered public-TRAIN-only metric diagnostic."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    except Exception as exc:  # pragma: no cover - environment report path
        return None, {"status": "unavailable", "reason": f"scikit-learn import: {exc}"}
    split = json.loads((OUT / "manifests/devplus_split.json").read_text())
    train_videos = set(int(x) for x in split["representation_train_videos"])
    z = np.load(OUT / "sources/full_tao_tracks.npz", allow_pickle=False) if (OUT / "sources/full_tao_tracks.npz").exists() else np.load(ROOT / "outputs/iclr27_phase6d/assets/full_tao_tracks.npz", allow_pickle=False)
    vids, labels = z["video_ids"].astype(np.int64), z["labels"].astype(np.int64)
    keep = np.asarray([int(v) in train_videos for v in vids], dtype=bool)
    keep &= z["is_known"].astype(bool)
    x = z["mean_feats"].astype(np.float32)[keep]
    y = labels[keep]
    if len(x) < 20 or len(np.unique(y)) < 3:
        return None, {"status": "unavailable", "reason": "insufficient public TRAIN tracks"}
    # Fixed, non-tuned dimensionality and solver.  DEV+ labels are not passed
    # to either fit call.
    pca_dim = min(64, x.shape[1], len(x) - 1)
    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=SEED)
    xp = pca.fit_transform(x)
    lda_dim = min(32, len(np.unique(y)) - 1, xp.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=lda_dim, solver="svd")
    lda.fit(xp, y)
    meta = {
        "status": "fit_public_train_only",
        "public_train_tracks": int(len(x)),
        "public_train_categories": int(len(np.unique(y))),
        "pca_components": int(pca_dim),
        "lda_components": int(lda_dim),
        "recipe": "PCA(64, randomized, seed=20260823) -> LDA(32 or rank)",
        "category_label_used": True,
        "devplus_labels_used_for_fit": False,
    }

    def transform(v):
        return norm(lda.transform(pca.transform(v)).astype(np.float32))

    return transform, meta


def main():
    rows = load_manifest()
    pooled, continuity = load_features(rows)
    labels = np.asarray([int(r["category_id"]) for r in rows], dtype=np.int64)
    videos = np.asarray([int(r["video_id"]) for r in rows], dtype=np.int64)
    out = {
        "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
        "q1_used": False,
        "view": "DEV+ GT-box diagnostic only; no proposal replay",
        "oracle": {
            **action_oracle(rows),
            "known_support": known_support_oracle(),
            "status": "measured_illegal_upper_bound",
            "oracle_label_used": True,
            "category_label_used": True,
            "future_frames_used": False,
            "physical_id_used_as_feature": False,
            "known_occurrence_accuracy": None,
            "known_occurrence_denominator": 0,
            "known_note": "DEV+ stream contains selected novel categories only; primary known routing is not evaluated without a proposal stream.",
        },
        "instance_only": {
            "status": "measured_gt_box_diagnostic",
            "representation": "normalized mean consecutive DINOv2 feature change; no category/cross-instance loss",
            "category_label_used": False,
            "future_frames_used": False,
            "physical_id_used_as_feature": False,
            "retrieval": track_metrics(continuity, labels, videos),
            "causal_nearest_past": retrieval_action_metrics(continuity, rows),
        },
    }
    transform, sup_meta = fit_supervised_projection()
    if transform is None:
        out["supervised_diagnostic"] = {**sup_meta, "category_label_used": True}
    else:
        projected = transform(pooled)
        out["supervised_diagnostic"] = {
            **sup_meta,
            "status": "measured_category_disjoint_gt_box_diagnostic",
            "retrieval": track_metrics(projected, labels, videos),
            "causal_nearest_past": retrieval_action_metrics(projected, rows),
            "future_frames_used": False,
            "physical_id_used_as_feature": False,
        }
    out["interpretation"] = {
        "oracle_expressivity": "positive if cross-video reuse denominator is nonzero and oracle accuracy is 1.0",
        "supervised_vs_instance": "compare cross-video retrieval and causal nearest-past reuse; supervised run is illegal positive control",
        "primary_gate_status": "not_run: proposal-aligned TrackOCD stream unavailable",
    }
    path = OUT / "eval/learnability_controls.json"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2))
    tmp.replace(path)
    summary = {
        "oracle": {k: out["oracle"].get(k) for k in ("devplus_novel_tracks", "novel_reuse_denominator", "cross_video_reuse_denominator", "novel_reuse_accuracy", "cross_video_reuse_accuracy")},
        "instance_cross_video_r1": out["instance_only"]["retrieval"]["cross_video_recall_at_1"],
        "supervised_status": out["supervised_diagnostic"].get("status"),
        "supervised_cross_video_r1": out["supervised_diagnostic"].get("retrieval", {}).get("cross_video_recall_at_1"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
