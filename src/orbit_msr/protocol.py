"""Shared IO and stats for ORBIT-MSR (memory-scale robust)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from src.orbit.protocol import (
    load_frame_features,
    load_mean_features,
    load_train_labels,
    meta_classes,
    ROOT,
)


def frozen_known_protos(class_ids):
    feats = load_mean_features("train_known_mean")
    labels = load_train_labels()
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float32))
    counts = defaultdict(int)
    for sid, c in labels.items():
        if c in class_ids and sid in feats:
            sums[c] += feats[sid]
            counts[c] += 1
    protos = {}
    for c, s in sums.items():
        v = s / counts[c]
        protos[int(c)] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def known_stats(z, P_known, radii, known_ids=None, anchor_best=None,
                best_n=-1.0, second_n=-1.0, margin_n=0.0, dist_n=1.0,
                rel=1.0, track_len=8, n_novel=0, include_anchor=False):
    ks = P_known @ z
    if ks.shape[0]:
        best_k = float(ks.max())
        order = np.argsort(ks)[::-1]
        second_k = float(ks[order[1]]) if ks.shape[0] >= 2 else best_k
        margin_k = best_k - second_k
        cid = int(known_ids[order[0]]) if known_ids is not None else int(order[0])
        r_k = float(radii.get(cid, 1.0))
        dist_k = (1.0 - best_k) / max(r_k, 1e-6)
    else:
        best_k = second_k = -1.0
        margin_k = 0.0
        dist_k = 1.0
    out = [best_k, second_k, margin_k, dist_k]
    if include_anchor:
        out.append(anchor_best if anchor_best is not None else -1.0)
    out += [best_n, second_n, margin_n, dist_n,
            rel, float(track_len) / 40.0, float(n_novel) / 300.0]
    return out


def novel_stats(z, P_novel, novel_counts, novel_radii, novel_ids=None,
                best_k=-1.0, margin_k=0.0, rel=1.0, track_len=8, n_novel=0,
                age_norm=0.0, mem_scale_norm=False):
    ns = P_novel @ z
    if ns.shape[0]:
        best_n = float(ns.max())
        order = np.argsort(ns)[::-1]
        second_n = float(ns[order[1]]) if ns.shape[0] >= 2 else best_n
        margin_n = best_n - second_n
        nid = int(novel_ids[order[0]]) if novel_ids is not None else int(order[0])
        r_n = float(novel_radii.get(nid, 0.3))
        dist_n = (1.0 - best_n) / max(r_n, 1e-6)
        support = float(np.log1p(novel_counts.get(nid, 0)))
        support_norm = support / float(np.log1p(300.0))
    else:
        best_n = second_n = -1.0
        margin_n = 0.0
        dist_n = 1.0
        support_norm = 0.0
    out = [best_n, second_n, margin_n, dist_n, support_norm,
           min(age_norm, 1.0), best_k, margin_k,
           rel, float(track_len) / 40.0, float(n_novel) / 300.0]
    if mem_scale_norm:
        # explicit memory-scale / hubness correction feature
        mem_scale = float(np.log1p(n_novel) / np.log1p(300.0))
        dist_n_scaled = dist_n / max(1.0, np.log2(1 + n_novel) / np.log2(65.0))
        out += [mem_scale, dist_n_scaled]
    return out


def stats_to_tensor(stats, device="cpu"):
    return torch.as_tensor([stats], dtype=torch.float32, device=device)


def build_meta_proxy_rows(max_known=600, seed=1027):
    train_feats = load_frame_features("train_known_mean")
    labels = load_train_labels()
    meta_tr = meta_classes("meta_train_classes")
    meta_dev = meta_classes("meta_dev_classes")
    dev_ids = sorted(sid for sid, c in labels.items() if c in meta_dev and sid in train_feats)
    known_ids = sorted(sid for sid, c in labels.items() if c in meta_tr and sid in train_feats)
    rng = np.random.RandomState(seed)
    rng.shuffle(known_ids)
    known_ids = known_ids[:max_known]
    gt_by_sid = {}
    rows = []
    n = max(len(dev_ids), len(known_ids))
    for i in range(n):
        if i < len(dev_ids):
            sid = dev_ids[i]
            rows.append({"sample_id": sid})
            gt_by_sid[sid] = {"sample_id": sid,
                              "ground_truth_category_id": labels[sid],
                              "protocol_role": "novel"}
        if i < len(known_ids):
            sid = known_ids[i]
            rows.append({"sample_id": sid})
            gt_by_sid[sid] = {"sample_id": sid,
                              "ground_truth_category_id": labels[sid],
                              "protocol_role": "supported_known"}
    return rows, gt_by_sid, train_feats, labels
