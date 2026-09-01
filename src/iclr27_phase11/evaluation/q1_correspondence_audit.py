"""Audit Q1 novel correspondence opportunities and frozen representations.

The audit deliberately reports two populations:

* all private-GT novel tracks in the locked Q1 videos (post-hoc only); and
* the subset actually covered by the corrected DSCT proposals and therefore
  visible to the strict evaluator.

GT is used only after feature extraction to annotate the report.  No value
from this file is used by an online decision or by training.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
Q1_PROPOSALS = ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv"
OUT = ROOT / "outputs/iclr27_phase11/eval/q1_correspondence_audit.json"
TAO_VAL = ROOT / "data/raw/tao/annotations/validation.json"
CLIP_CACHE = ROOT / "data/caches/features/clip/gt_tracks_mean"
DINO_CACHE = ROOT / "data/caches/features/dinov2/gt_tracks_mean"

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes  # noqa: E402
from src.iclr27_phase4s.protocol import (  # noqa: E402
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase6c.model.tse import TSE  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import (  # noqa: E402
    load_tse,
    project,
)
from src.iclr27_phase8a.model.adapter import CausalTrajectoryAdapter  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), 1e-12)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def summary(values) -> dict:
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "p10": None,
                "p90": None, "min": None, "max": None}
    return {
        "n": int(len(a)), "mean": float(a.mean()),
        "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)), "min": float(a.min()),
        "max": float(a.max()),
    }


def load_caches(records: list[dict]):
    out = {}
    for r in records:
        sid = r["sample_id"]
        cp = CLIP_CACHE / f"{sid}.json"
        dp = DINO_CACHE / f"{sid}.json"
        if not cp.exists() or not dp.exists():
            raise FileNotFoundError(f"missing cached foundation feature for {sid}")
        c = json.loads(cp.read_text())
        d = json.loads(dp.read_text())
        out[sid] = {
            "clip_mean": l2(np.asarray(c["mean_embedding"], dtype=np.float32)),
            "clip_frames": l2(np.asarray(c["frame_embeddings"], dtype=np.float32)),
            "dino_mean": l2(np.asarray(d["mean_embedding"], dtype=np.float32)),
            "dino_frames": l2(np.asarray(d["frame_embeddings"], dtype=np.float32)),
            "clip_dim": int(len(c["mean_embedding"])),
            "dino_dim": int(len(d["mean_embedding"])),
        }
    return out


def make_tse_b_features(cache: dict[str, dict]):
    """Return post-hoc TSE and B vectors for cached GT track sequences."""
    device = torch.device("cpu")
    tse, _, _ = load_tse(device)
    bck = torch.load(
        ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth",
        map_location=device, weights_only=False)
    b_args = bck.get("args", {})
    dim = int(b_args.get("dim", 128))
    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=bool(b_args.get("frame_level", False))).to(device)
    adapter.load_state_dict(bck["adapter"])
    adapter.eval()
    out = {}
    with torch.no_grad():
        for sid, item in cache.items():
            raw = item["dino_frames"].astype(np.float32)
            z = project(device, tse, raw)
            zt = l2(z.mean(axis=0))
            st = adapter.new_state()
            bh = []
            for row in z:
                h, st = adapter(torch.from_numpy(row).to(device).unsqueeze(0), st)
                bh.append(h[0].cpu().numpy().astype(np.float32))
            out[sid] = {"tse": zt, "phase8a_b": l2(np.asarray(bh).mean(axis=0))}
    return out


def geometry(records: list[dict], vectors: dict[str, np.ndarray]) -> dict:
    labels = np.asarray([int(r["category"]) for r in records], dtype=np.int64)
    videos = np.asarray([int(r["video_id"]) for r in records], dtype=np.int64)
    ids = [r["sample_id"] for r in records]
    x = l2(np.stack([vectors[s] for s in ids]))
    same, inter, pairs = [], [], []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            dist = float(1.0 - np.dot(x[i], x[j]))
            same_cat = bool(labels[i] == labels[j])
            cross_video = bool(videos[i] != videos[j])
            if same_cat:
                same.append(dist)
                pairs.append({
                    "category": int(labels[i]), "left": ids[i], "right": ids[j],
                    "distance": dist, "cross_video": cross_video,
                })
            else:
                inter.append(dist)
    nn_correct = 0
    cross_correct = 0
    cross_total = 0
    nn_dist = []
    for i in range(len(ids)):
        cand = [j for j in range(len(ids)) if j != i]
        if not cand:
            continue
        j = min(cand, key=lambda q: float(1.0 - np.dot(x[i], x[q])))
        nn_dist.append(float(1.0 - np.dot(x[i], x[j])))
        nn_correct += int(labels[i] == labels[j])
        cross = [q for q in cand if videos[q] != videos[i]]
        if cross:
            q = min(cross, key=lambda z: float(1.0 - np.dot(x[i], x[z])))
            cross_total += 1
            cross_correct += int(labels[i] == labels[q])
    km = None
    if len(x) >= 2:
        k = min(len(np.unique(labels)), len(x))
        pred = KMeans(n_clusters=k, n_init=20, random_state=1027).fit(x).labels_
        km = {"n_clusters": int(k),
              "nmi": float(normalized_mutual_info_score(labels, pred)),
              "ari": float(adjusted_rand_score(labels, pred))}
    return {
        "n_tracks": len(ids), "n_categories": int(len(np.unique(labels))),
        "same_category_pairs": len(same),
        "same_category_cross_video_pairs": int(sum(p["cross_video"] for p in pairs)),
        "same_category_distance": summary(same),
        "different_category_distance": summary(inter),
        "inter_minus_intra": (float(np.mean(inter) - np.mean(same))
                               if same and inter else None),
        "nearest_neighbor_accuracy": float(nn_correct / max(len(ids), 1)),
        "cross_video_nearest_neighbor_accuracy": float(
            cross_correct / max(cross_total, 1)),
        "cross_video_nearest_neighbor_n": int(cross_total),
        "nearest_neighbor_distance": summary(nn_dist),
        "kmeans": km,
        "pairs": pairs,
    }


def category_table(records: list[dict], aligned_ids: set[str]) -> list[dict]:
    by = defaultdict(list)
    for r in records:
        by[int(r["category"])].append(r)
    out = []
    for cat in sorted(by):
        rs = by[cat]
        same_pairs = []
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                same_pairs.append({
                    "cross_video": int(rs[i]["video_id"]) != int(rs[j]["video_id"]),
                    "left": rs[i]["sample_id"], "right": rs[j]["sample_id"],
                })
        out.append({
            "category": cat, "tracks": len(rs),
            "videos": sorted({int(r["video_id"]) for r in rs}),
            "aligned_tracks": sum(r["sample_id"] in aligned_ids for r in rs),
            "aligned_videos": sorted({int(r["video_id"]) for r in rs
                                       if r["sample_id"] in aligned_ids}),
            "same_category_pairs": same_pairs,
        })
    return out


def main():
    stream, labels = load_gt_tracks_dev()
    labels = {r["sample_id"]: labels[r["sample_id"]] for r in stream}
    gt_novel = [
        {"sample_id": r["sample_id"], "video_id": int(r["video_id"]),
         "track_id": int(r["track_id"]),
         "category": int(labels[r["sample_id"]]["ground_truth_category_id"]),
         "length": len(r["frame_ids"]), "areas": r.get("areas", []),
         "image_paths": r["image_paths"], "boxes": r["boxes_xyxy"]}
        for r in stream if labels[r["sample_id"]]["protocol_role"] == "novel"
    ]
    rows = load_proposals(Q1_PROPOSALS)
    mapping = align_pred_to_gt(group_tracks(rows), gt_track_boxes(stream))
    aligned_ids = {sid for sid in mapping.values()
                   if sid in {r["sample_id"] for r in gt_novel}}
    aligned = [r for r in gt_novel if r["sample_id"] in aligned_ids]
    cache = load_caches(gt_novel)
    tse_b = make_tse_b_features(cache)

    ann = json.loads(TAO_VAL.read_text())
    by_path = {im["file_name"]: im for im in ann["images"]}
    for r in gt_novel:
        abs_area, frac_area, wr, hr = [], [], [], []
        for path, box in zip(r["image_paths"], r["boxes"]):
            im = by_path.get(path)
            if im is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            abs_area.append(area)
            frac_area.append(area / max(float(im["width"] * im["height"]), 1.0))
            wr.append((x2 - x1) / max(float(im["width"]), 1.0))
            hr.append((y2 - y1) / max(float(im["height"]), 1.0))
        r["object_area_pixels_mean"] = float(np.mean(abs_area)) if abs_area else None
        r["object_area_fraction_mean"] = float(np.mean(frac_area)) if frac_area else None
        r["bbox_width_fraction_mean"] = float(np.mean(wr)) if wr else None
        r["bbox_height_fraction_mean"] = float(np.mean(hr)) if hr else None
        r.pop("areas", None); r.pop("image_paths", None); r.pop("boxes", None)

    feature_vectors = {
        "dino_v2": {sid: cache[sid]["dino_mean"] for sid in cache},
        "clip_image_vit_b32": {sid: cache[sid]["clip_mean"] for sid in cache},
        "tse": {sid: tse_b[sid]["tse"] for sid in cache},
        "phase8a_b": {sid: tse_b[sid]["phase8a_b"] for sid in cache},
    }
    geometry_all = {name: geometry(gt_novel, vec)
                    for name, vec in feature_vectors.items()}
    geometry_aligned = {name: geometry(aligned, vec)
                        for name, vec in feature_vectors.items()}

    def length_size(rs):
        return {
            "track_length": summary([r["length"] for r in rs]),
            "object_area_pixels_mean_per_track": summary(
                [r["object_area_pixels_mean"] for r in rs
                 if r["object_area_pixels_mean"] is not None]),
            "object_area_fraction_mean_per_track": summary(
                [r["object_area_fraction_mean"] for r in rs
                 if r["object_area_fraction_mean"] is not None]),
            "bbox_width_fraction_mean_per_track": summary(
                [r["bbox_width_fraction_mean"] for r in rs
                 if r["bbox_width_fraction_mean"] is not None]),
            "bbox_height_fraction_mean_per_track": summary(
                [r["bbox_height_fraction_mean"] for r in rs
                 if r["bbox_height_fraction_mean"] is not None]),
        }

    full_pairs = sum(len(x["same_category_pairs"]) for x in category_table(gt_novel, aligned_ids))
    full_cross_pairs = sum(
        int(p["cross_video"])
        for x in category_table(gt_novel, aligned_ids)
        for p in x["same_category_pairs"])
    aligned_pairs = sum(len(x["same_category_pairs"])
                        for x in category_table(aligned, aligned_ids))
    aligned_cross_pairs = sum(
        int(p["cross_video"])
        for x in category_table(aligned, aligned_ids)
        for p in x["same_category_pairs"])

    out = {
        "protocol": {
            "q1_rows": len(rows), "q1_proposal_tracks": len(group_tracks(rows)),
            "gt_dev_tracks": len(labels), "gt_used_posthoc_only": True,
            "proposal_to_gt_mapping_tracks": len(mapping),
            "feature_cache_note": "DINOv2 and OpenAI CLIP ViT-B/32 GT-track caches; TSE/B are post-hoc causal-prefix transforms",
        },
        "opportunity_audit": {
            "all_gt_novel": {
                "tracks": len(gt_novel),
                "categories": len({r["category"] for r in gt_novel}),
                "same_category_pairs": full_pairs,
                "same_category_cross_video_pairs": full_cross_pairs,
                "track_and_size": length_size(gt_novel),
            },
            "strict_evaluable_aligned_novel": {
                "tracks": len(aligned),
                "categories": len({r["category"] for r in aligned}),
                "same_category_pairs": aligned_pairs,
                "same_category_cross_video_pairs": aligned_cross_pairs,
                "track_and_size": length_size(aligned),
            },
            "coverage": {
                "novel_track_recall": len(aligned) / max(len(gt_novel), 1),
                "novel_category_recall": len({r["category"] for r in aligned}) /
                max(len({r["category"] for r in gt_novel}), 1),
                "same_category_pair_recall": aligned_pairs / max(full_pairs, 1),
                "cross_video_pair_recall": aligned_cross_pairs / max(full_cross_pairs, 1),
            },
            "per_category": category_table(gt_novel, aligned_ids),
            "aligned_track_ids": sorted(aligned_ids),
        },
        "gt_cached_feature_geometry": {
            "all_gt_novel": geometry_all,
            "strict_evaluable_aligned_novel": geometry_aligned,
        },
        "existing_phase10_predicted_track_geometry": {
            "source": "outputs/iclr27_phase10/eval/representation_diagnosis_hybrid.json",
            "note": "The Phase-10 numbers use proposal-track features and are retained for exact baseline comparability.",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({
        "all_gt_novel": out["opportunity_audit"]["all_gt_novel"],
        "aligned": out["opportunity_audit"]["strict_evaluable_aligned_novel"],
        "coverage": out["opportunity_audit"]["coverage"],
        "geometry_all_keys": list(geometry_all),
        "geometry_aligned_keys": list(geometry_aligned),
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
