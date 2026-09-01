"""Post-hoc representation geometry audit for Phase 6C TSE.

Compares raw DINOv2 track means vs TSE embeddings on:
  - known train leave-one-out prototype accuracy
  - known intra/inter class cosine ratio
  - Q1 aligned known-track prototype accuracy
  - Q1 novel-track kNN purity (private GT used ONLY as post-hoc diagnostic,
    never for training or online decisions)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

from src.iclr27_phase6c.evaluation.online_semantic_eval import load_tse
from src.iclr27_phase6c.model.tse import TSE
from src.iclr27_phase4s.protocol import load_gt_tracks_dev, load_proposals
from src.iclr27_phase4s.frontend import align_pred_to_gt, gt_track_boxes
from src.iclr27_phase4s.protocol import group_tracks


def l2norm(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def knn_purity(X, labels, k=5, cross_video=None):
    sim = X @ X.T
    np.fill_diagonal(sim, -1)
    hits, total = 0, 0
    for i in range(len(X)):
        top = np.argsort(sim[i])[-k:][::-1]
        for j in top:
            if cross_video is not None and cross_video[i] == cross_video[j]:
                continue
            hits += int(labels[i] == labels[j])
            total += 1
    return hits / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/iclr27_phase6c/eval/geometry_main.json")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, anchors, known_ids = load_tse(ROOT / args.ckpt, dev)
    k = np.load(ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz")
    u = np.load(ROOT / "outputs/iclr27_phase6c/assets/unlabeled_tracks.npz")
    with torch.no_grad():
        zk = model.project(torch.from_numpy(k["mean_feats"].astype(np.float32)).to(dev)
                           ).cpu().numpy()
        zu = model.project(torch.from_numpy(u["mean_feats"].astype(np.float32)).to(dev)
                           ).cpu().numpy()
    zk, zu = l2norm(zk), l2norm(zu)
    rawk = l2norm(k["mean_feats"])
    labels = k["labels"]

    def loo_acc(Z):
        ids = np.unique(labels)
        correct = 0
        for i in range(len(Z)):
            proto = {}
            for c in ids:
                idx = np.where(labels == c)[0]
                idx = idx[idx != i]
                if len(idx) == 0:
                    continue
                proto[c] = l2norm(Z[idx].mean(axis=0))
            if not proto:
                continue
            best_c, best_s = None, -1.0
            for c, p in proto.items():
                s = float(Z[i] @ p)
                if s > best_s:
                    best_s, best_c = s, c
            correct += int(best_c == labels[i])
        return correct / len(Z)

    raw_loo = loo_acc(rawk)
    tse_loo = loo_acc(zk)

    def intra_inter(Z):
        intra, inter = [], []
        by_cat = defaultdict(list)
        for i, c in enumerate(labels):
            by_cat[int(c)].append(i)
        for c, idx in by_cat.items():
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    intra.append(float(Z[idx[a]] @ Z[idx[b]]))
        cats = list(by_cat)
        for a in range(0, len(Z), 50):
            for b in range(a + 1, len(Z), 50):
                if labels[a] != labels[b]:
                    inter.append(float(Z[a] @ Z[b]))
        return float(np.mean(intra)), float(np.mean(inter)), (
            float(np.mean(intra)) / max(float(np.mean(inter)), 1e-9))

    raw_ii = intra_inter(rawk)
    tse_ii = intra_inter(zk)

    # Q1 aligned diagnostics (private GT post-hoc only)
    q1 = {
        "known_track_proto_acc_raw": None,
        "known_track_proto_acc_tse": None,
        "novel_knn_purity_raw": None,
        "novel_knn_purity_tse": None,
        "n_novel_tracks": None,
    }
    try:
        phys = ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv"
        rows = load_proposals(phys)
        feats = np.load(ROOT / "outputs/iclr27_phase6b/q1/final_dsct/feats.npz")["feats"]
        tracks = group_tracks(rows)
        stream, labels_all = load_gt_tracks_dev()
        gb = gt_track_boxes(stream)
        mapping = align_pred_to_gt(tracks, gb)
        gt_labels = {r["sample_id"]: labels_all[r["sample_id"]] for r in stream}
        key_means_raw = {}
        key_means_tse = {}
        key_role = {}
        with torch.no_grad():
            zq = model.project(torch.from_numpy(feats.astype(np.float32)).to(dev)
                               ).cpu().numpy()
        zq = l2norm(zq)
        for key, idxs in tracks.items():
            if key not in mapping:
                continue
            raw = np.stack([feats[rows.index(r)] for r in idxs]).mean(axis=0)
            zt = np.stack([zq[rows.index(r)] for r in idxs]).mean(axis=0)
            key_means_raw[key] = l2norm(raw)
            key_means_tse[key] = l2norm(zt)
            key_role[key] = gt_labels[mapping[key]]["protocol_role"]
        known_keys = [key for key, role in key_role.items()
                      if role in ("supported_known", "zero_shot_known")]
        novel_keys = [key for key, role in key_role.items() if role == "novel"]

        # raw prototypes built from raw TRAIN class means; TSE prototypes
        # built from TSE TRAIN class means
        def build_tse_protos():
            proto = {}
            for c in np.unique(k["labels"]):
                proto[int(c)] = zk[k["labels"] == c].mean(axis=0)
            return proto

        def proto_acc_custom(keys, Z, proto):
            correct = 0
            for key in keys:
                best_c, best_s = None, -1.0
                for c, p in proto.items():
                    s = float(Z[key] @ l2norm(p))
                    if s > best_s:
                        best_s, best_c = s, c
                correct += int(best_c == int(
                    gt_labels[mapping[key]]["ground_truth_category_id"]))
            return correct / max(len(keys), 1)

        raw_protos = {}
        for c in np.unique(k["labels"]):
            raw_protos[int(c)] = rawk[k["labels"] == c].mean(axis=0)
        tse_protos = build_tse_protos()
        q1["known_track_proto_acc_raw"] = proto_acc_custom(
            known_keys, key_means_raw, raw_protos)
        q1["known_track_proto_acc_tse"] = proto_acc_custom(
            known_keys, key_means_tse, tse_protos)
        novel_cat = np.array([int(gt_labels[mapping[key]]["ground_truth_category_id"])
                              for key in novel_keys])
        q1["n_novel_tracks"] = len(novel_keys)
        if len(novel_keys) >= 2:
            Xr = np.stack([key_means_raw[key] for key in novel_keys])
            Xt = np.stack([key_means_tse[key] for key in novel_keys])
            q1["novel_knn_purity_raw"] = knn_purity(Xr, novel_cat, k=min(3, len(novel_keys) - 1))
            q1["novel_knn_purity_tse"] = knn_purity(Xt, novel_cat, k=min(3, len(novel_keys) - 1))
    except Exception as e:
        q1["error"] = repr(e)

    out = {
        "raw_loo_known_acc": raw_loo,
        "tse_loo_known_acc": tse_loo,
        "raw_intra_inter_ratio": raw_ii[2],
        "tse_intra_inter_ratio": tse_ii[2],
        "raw_intra": raw_ii[0],
        "raw_inter": raw_ii[1],
        "tse_intra": tse_ii[0],
        "tse_inter": tse_ii[1],
        "q1": q1,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
