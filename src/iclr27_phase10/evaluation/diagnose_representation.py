"""Post-hoc Q1 geometry diagnosis for Phase 10.

The script compares three representations on the *same frozen physical Q1
stream*:

1. raw DINOv2 bbox features;
2. frozen Phase-6C TSE frame features;
3. the frozen Phase-8A B causal adapter output (semantic trajectory feature).

Track embeddings are means of the frame embeddings, followed by L2
normalization.  This aggregation is used only for an offline diagnostic; no
embedding or label from the future is used by any online replay.  Private GT
is joined only after feature extraction to quantify novel-category geometry.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
Q1_VIDEOS = [88, 90, 122, 291, 334, 888, 931, 1159, 1232, 1276,
             1572, 1865, 2254, 2347, 2564, 2675, 2690, 2759, 2802, 2888]

from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes  # noqa: E402
from src.iclr27_phase4s.protocol import (  # noqa: E402
    group_tracks,
    load_gt_tracks_dev,
    load_proposals,
)
from src.iclr27_phase6c.model.tse import TSE  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import project  # noqa: E402
from src.iclr27_phase8a.model.adapter import CausalTrajectoryAdapter  # noqa: E402
from src.iclr27_phase10.model.hybrid import HybridTrajectoryEncoder  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), 1e-12)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def percentile_summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None,
                "p10": None, "p90": None}
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
    }


def pair_geometry(X: np.ndarray, labels: np.ndarray, videos: np.ndarray) -> dict:
    """Novel-track geometry and unsupervised cluster diagnostics."""
    X = l2(X)
    n = len(X)
    intra, inter = [], []
    same_video_intra, cross_video_intra = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(1.0 - np.dot(X[i], X[j]))
            if int(labels[i]) == int(labels[j]):
                intra.append(d)
                if int(videos[i]) == int(videos[j]):
                    same_video_intra.append(d)
                else:
                    cross_video_intra.append(d)
            else:
                inter.append(d)

    nn_correct = 0
    nn_cross_video_correct = 0
    nn_cross_video_total = 0
    nn_dist = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        if not candidates:
            continue
        j = min(candidates, key=lambda q: float(1.0 - np.dot(X[i], X[q])))
        d = float(1.0 - np.dot(X[i], X[j]))
        nn_dist.append(d)
        nn_correct += int(labels[i] == labels[j])
        cross = [q for q in candidates if int(videos[q]) != int(videos[i])]
        if cross:
            q = min(cross, key=lambda z: float(1.0 - np.dot(X[i], X[z])))
            nn_cross_video_total += 1
            nn_cross_video_correct += int(labels[i] == labels[q])

    n_cats = int(len(np.unique(labels)))
    kmeans = None
    if n >= 2:
        # K is fixed from the post-hoc number of novel categories, not tuned
        # against the representation.  Labels are never used by replay.
        km = KMeans(n_clusters=min(n_cats, n), n_init=20,
                    random_state=1027).fit(X)
        pred = km.labels_
        kmeans = {
            "n_clusters": int(min(n_cats, n)),
            "nmi": float(normalized_mutual_info_score(labels, pred)),
            "ari": float(adjusted_rand_score(labels, pred)),
        }
    return {
        "n_tracks": int(n),
        "n_categories": n_cats,
        "category_counts": {
            str(int(c)): int((labels == c).sum()) for c in np.unique(labels)
        },
        "same_category_cosine_distance": percentile_summary(intra),
        "different_category_cosine_distance": percentile_summary(inter),
        "same_category_cosine_similarity": percentile_summary(
            [1.0 - x for x in intra]),
        "different_category_cosine_similarity": percentile_summary(
            [1.0 - x for x in inter]),
        "intra_inter_distance_ratio": (
            float(np.mean(intra) / max(np.mean(inter), 1e-12))
            if intra and inter else None),
        "inter_minus_intra_distance": (
            float(np.mean(inter) - np.mean(intra))
            if intra and inter else None),
        "nearest_neighbor_accuracy": float(nn_correct / max(n, 1)),
        "nearest_neighbor_distance": percentile_summary(nn_dist),
        "cross_video_nearest_neighbor_accuracy": float(
            nn_cross_video_correct / max(nn_cross_video_total, 1)),
        "cross_video_nearest_neighbor_n": int(nn_cross_video_total),
        "kmeans": kmeans,
        "note": "all GT usage is post-hoc geometry evaluation only",
    }


def load_b_adapter(device: torch.device):
    ckpt = ROOT / "outputs/iclr27_phase8a/training/b_pilot_scaled/best.pth"
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    args = ck.get("args", {})
    dim = int(args.get("dim", 128))
    adapter = CausalTrajectoryAdapter(
        dim=dim, rho_init=0.0, sigma2=1.0,
        frame_level=bool(args.get("frame_level", False))).to(device)
    adapter.load_state_dict(ck["adapter"])
    adapter.eval()
    return adapter, str(ckpt)


def causal_b_features(z: np.ndarray, rows: list[dict],
                      device: torch.device,
                      adapter: CausalTrajectoryAdapter) -> np.ndarray:
    """Run the frozen B adapter in the exact chronological prefix order."""
    out = np.zeros((len(rows), adapter.dim), dtype=np.float32)
    track_state = {}
    chrono = sorted(
        range(len(rows)),
        key=lambda i: (int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
                       int(rows[i].get("proposal_local_id") or 0),
                       int(rows[i]["track_id"])),
    )
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            prev = track_state.get(key)
            if prev is None:
                prev = adapter.new_state()
            x = torch.from_numpy(z[i]).to(device).unsqueeze(0)
            h, state = adapter(x, prev)
            track_state[key] = state.detach()
            out[i] = h[0].cpu().numpy().astype(np.float32)
    return l2(out)


def causal_hybrid_features(z: np.ndarray, rows: list[dict],
                           device: torch.device, checkpoint: str) -> np.ndarray:
    ck = torch.load(ROOT / checkpoint, map_location=device, weights_only=False)
    a = ck.get("args", {})
    model = HybridTrajectoryEncoder(dim=z.shape[1], hidden=int(a.get("hidden", 128)),
                                    out_dim=z.shape[1]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    out = np.zeros((len(rows), z.shape[1]), dtype=np.float32)
    states = {}
    chrono = sorted(range(len(rows)), key=lambda i: (
        int(rows[i]["video_id"]), int(rows[i]["frame_id"]),
        int(rows[i].get("proposal_local_id") or 0), int(rows[i]["track_id"])))
    with torch.no_grad():
        for i in chrono:
            key = (int(rows[i]["video_id"]), int(rows[i]["track_id"]))
            st = states.get(key)
            if st is None:
                st = model.new_state(device=device)
            h, st = model.step(torch.from_numpy(z[i]).to(device).unsqueeze(0), st)
            states[key] = st
            out[i] = h[0].cpu().numpy().astype(np.float32)
    return l2(out)


def track_means(rows: list[dict], frame_features: np.ndarray) -> dict:
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[(int(r["video_id"]), int(r["track_id"]))].append(i)
    return {k: l2(frame_features[v].mean(axis=0)) for k, v in idx.items()}


def known_proto_accuracy(track_vectors: dict, keys: list[tuple[int, int]],
                         gt_labels: dict, mapping: dict,
                         train_vectors: np.ndarray, train_labels: np.ndarray) -> float:
    protos = {}
    for c in np.unique(train_labels):
        protos[int(c)] = l2(train_vectors[train_labels == c].mean(axis=0))
    correct = 0
    for key in keys:
        pred = max(protos, key=lambda c: float(np.dot(track_vectors[key], protos[c])))
        true = int(gt_labels[mapping[key]]["ground_truth_category_id"])
        correct += int(pred == true)
    return float(correct / max(len(keys), 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hybrid-checkpoint", default=None)
    ap.add_argument("--out", default="outputs/iclr27_phase10/eval/representation_diagnosis.json")
    args = ap.parse_args()
    device = torch.device(args.device)

    proposal_path = ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv"
    feat_path = ROOT / "outputs/iclr27_phase6b/q1/final_dsct/feats.npz"
    rows = load_proposals(proposal_path)
    raw = np.load(feat_path)["feats"].astype(np.float32)
    if len(rows) != len(raw):
        raise RuntimeError(f"row/feature mismatch: {len(rows)} vs {len(raw)}")

    tracks = group_tracks(rows)
    stream, labels_all = load_gt_tracks_dev()
    mapping = align_pred_to_gt(tracks, gt_track_boxes(stream))
    labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}

    # Frozen TSE frame embedding.
    state = torch.load(ROOT / "outputs/iclr27_phase6c/training/tse_main/checkpoint.pth",
                       map_location=device, weights_only=False)
    tse = TSE().to(device)
    tse.load_pca(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    tse.load_state_dict(state["model"])
    tse.eval()
    tse_z = project(device, tse, raw)

    # Frozen Architecture-B semantic trajectory embedding.
    adapter, b_ckpt = load_b_adapter(device)
    b_h = causal_b_features(tse_z, rows, device, adapter)

    raw_n = l2(raw)
    tse_n = l2(tse_z)
    raw_tracks = track_means(rows, raw_n)
    tse_tracks = track_means(rows, tse_n)
    b_tracks = track_means(rows, b_h)

    role_by_key = {k: labels[sid] for k, sid in mapping.items()}
    novel_keys = [k for k, lab in role_by_key.items()
                  if lab["protocol_role"] == "novel"]
    known_keys = [k for k, lab in role_by_key.items()
                  if lab["protocol_role"] in ("supported_known", "zero_shot_known")]
    novel_labels = np.asarray([
        int(role_by_key[k]["ground_truth_category_id"]) for k in novel_keys],
        dtype=np.int64)
    novel_videos = np.asarray([int(k[0]) for k in novel_keys], dtype=np.int64)

    geometry = {}
    for name, vectors in (("dino_bbox", raw_tracks),
                          ("tse", tse_tracks),
                          ("phase8a_b", b_tracks)):
        X = np.stack([vectors[k] for k in novel_keys])
        geometry[name] = pair_geometry(X, novel_labels, novel_videos)

    hybrid_tracks = None
    if args.hybrid_checkpoint:
        hybrid_h = causal_hybrid_features(tse_z, rows, device,
                                           args.hybrid_checkpoint)
        hybrid_tracks = track_means(rows, hybrid_h)
        geometry["phase10_hybrid"] = pair_geometry(
            np.stack([hybrid_tracks[k] for k in novel_keys]),
            novel_labels, novel_videos)

    # Known-track sanity is useful to distinguish a globally bad feature from
    # a specifically missing novel invariance signal.
    train = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    with torch.no_grad():
        train_tse = project(device, tse,
                            train["mean_feats"].astype(np.float32))
    train_raw = l2(train["mean_feats"].astype(np.float32))
    train_tse = l2(train_tse)
    known_acc = {
        "dino_bbox": known_proto_accuracy(raw_tracks, known_keys, labels,
                                           mapping, train_raw,
                                           train["labels"]),
        "tse": known_proto_accuracy(tse_tracks, known_keys, labels, mapping,
                                     train_tse, train["labels"]),
        "phase8a_b": known_proto_accuracy(
            b_tracks, known_keys, labels, mapping, train_tse, train["labels"]),
    }

    out = {
        "protocol": {
            "q1_videos": Q1_VIDEOS,
            "n_rows": len(rows),
            "n_tracks": len(tracks),
            "n_aligned_tracks": len(mapping),
            "n_novel_tracks": len(novel_keys),
            "n_novel_categories": int(len(np.unique(novel_labels))),
            "novel_track_keys": [list(k) for k in novel_keys],
            "novel_categories": [int(x) for x in novel_labels],
            "gt_used_only_posthoc": True,
        },
        "representations": {
            "dino_bbox": {
                "source": str(feat_path),
                "dim": int(raw.shape[1]),
                "aggregation": "L2-normalized per-track mean",
            },
            "tse": {
                "source": "outputs/iclr27_phase6c/training/tse_main/checkpoint.pth",
                "dim": int(tse_z.shape[1]),
                "aggregation": "L2-normalized per-track mean",
            },
            "phase8a_b": {
                "source": b_ckpt,
                "dim": int(b_h.shape[1]),
                "aggregation": "causal adapter h, then L2-normalized per-track mean",
            },
        },
        "novel_geometry": geometry,
        "known_track_proto_accuracy": known_acc,
        "interpretation": {
            "diagnostic_question": (
                "If same-category novel tracks are close, decision/memory is the "
                "bottleneck; if separated, representation is the bottleneck."),
            "recommended_reading": (
                "Use novel inter/intra distance gap and nearest-neighbor accuracy "
                "before any Phase-10 training or online rule design."),
        },
    }
    if hybrid_tracks is not None:
        with torch.no_grad():
            # Reuse the frozen training TSE features as the hybrid input.
            kraw = train["frame_feats"].astype(np.float32)
            kz = project(device, tse, kraw.reshape(-1, kraw.shape[-1]))
            kz = kz.reshape(kraw.shape[0], kraw.shape[1], -1)
            hck = torch.load(ROOT / args.hybrid_checkpoint,
                             map_location=device, weights_only=False)
            ha = hck.get("args", {})
            hm = HybridTrajectoryEncoder(dim=kz.shape[-1],
                                         hidden=int(ha.get("hidden", 128)),
                                         out_dim=kz.shape[-1]).to(device)
            hm.load_state_dict(hck["model"])
            hm.eval()
            hf, _ = hm(torch.from_numpy(kz).to(device),
                       torch.from_numpy(train["frame_mask"]).to(device))
            hf = hf.cpu().numpy().astype(np.float32)
        hprotos = {}
        for c in np.unique(train["labels"]):
            hprotos[int(c)] = l2(hf[train["labels"] == c].mean(axis=0))
        pred = []
        for key in known_keys:
            pred.append(max(hprotos, key=lambda c: float(
                np.dot(hybrid_tracks[key], hprotos[c]))))
        true = [int(labels[mapping[k]]["ground_truth_category_id"])
                for k in known_keys]
        out["known_track_proto_accuracy"]["phase10_hybrid"] = float(
            np.mean(np.asarray(pred) == np.asarray(true)))
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
