"""Trajectory Semantic Encoder (TSE) for Phase 6C.

Architecture:
  frame DINOv2 (frozen, L2) -> base linear (PCA-initialized) + small
  residual MLP -> L2 frame embedding
  track embedding = mean of frame embeddings (L2)
  48 known anchors (initialized from PCA-projected train class means)

Objectives:
  - known cross-entropy + anchor attraction
  - same-track temporal InfoNCE (frames pulled to their track mean)
  - cross-track mutual-nearest-neighbor attraction on unlabeled trajectories
  - anchor preservation: keep anchors and track means close to the frozen
    PCA-projected foundation geometry
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm_t(x, dim=-1):
    return F.normalize(x, dim=dim, eps=1e-12)


class TSE(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=128, residual_scale=0.1):
        super().__init__()
        self.base = nn.Linear(in_dim, out_dim, bias=True)
        self.residual = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
        self.residual_scale = residual_scale
        # zero-init the final residual layer so the encoder starts at PCA
        nn.init.zeros_(self.residual[2].weight)
        nn.init.zeros_(self.residual[2].bias)
        self.out_dim = out_dim

    def load_pca(self, pca_path: Path):
        p = np.load(pca_path)
        comp = torch.from_numpy(p["components"].astype(np.float32))
        mean = torch.from_numpy(p["mean"].astype(np.float32))
        # PCA projection x -> (x - mean) @ components.T
        # Implement as a linear layer: y = W x + b
        with torch.no_grad():
            comp = comp.to(self.base.weight.device)
            mean = mean.to(self.base.weight.device)
            self.base.weight.copy_(comp)
            self.base.bias.copy_(-comp @ mean)
        self.register_buffer("pca_components", comp)
        self.register_buffer("pca_mean", mean)

    def project(self, x):
        """x: (..., in_dim) L2-normalized raw DINOv2 features -> L2 emb."""
        h = self.base(x)
        r = self.residual(h) * self.residual_scale
        return l2norm_t(h + r)

    def track_embed(self, frame_feats, mask=None):
        """frame_feats: (B,T,in); mask: (B,T) uint8/float."""
        zf = self.project(frame_feats)
        if mask is None:
            z = zf.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).float()
            z = (zf * m).sum(dim=1) / (m.sum(dim=1) + 1e-9)
        return l2norm_t(z), zf

    def pca_proj(self, x):
        return l2norm_t(F.linear(x, self.pca_components, -self.pca_components @ self.pca_mean))


class KnownAnchors(nn.Module):
    def __init__(self, known_ids, out_dim=128, init_feats=None):
        """init_feats: (K, out_dim) PCA-projected class means (optional)."""
        super().__init__()
        self.register_buffer("known_ids", torch.as_tensor(known_ids, dtype=torch.int64))
        p = torch.randn(len(known_ids), out_dim)
        if init_feats is not None:
            p = torch.from_numpy(init_feats.astype(np.float32))
        self.protos = nn.Parameter(l2norm_t(p))

    def normalized(self):
        return l2norm_t(self.protos)


def anchor_preservation_loss(anchors, pca_class_means):
    """Keep learnable anchors near the frozen PCA class centers."""
    return F.mse_loss(anchors, torch.from_numpy(pca_class_means).to(anchors.device))


def temporal_info_nce(zf, zt, anchors, tau=0.07):
    """Frames of each track are positives against its track mean.

    zf: (B,T,D), zt: (B,D), anchors: (K,D).
    """
    B, T, D = zf.shape
    # positives: own track mean per frame
    pos = (zf * zt.unsqueeze(1)).sum(-1) / tau  # (B,T)
    # negatives: all other track means + all anchors
    neg = zf @ torch.cat([zt, anchors], dim=0).t() / tau  # (B,T,B+K)
    # mask own track (index 0 in track block; anchors start at B)
    ar = torch.arange(B, device=zf.device)
    neg[:, :, ar] = -1e9
    logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1)  # (B,T,1+B+K)
    labels = torch.zeros(B, T, dtype=torch.long, device=zf.device)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))


def mnn_attraction(z, k=8):
    """Mutual-nearest-neighbor attraction on unlabeled track embeddings.

    For every track, the target is the mean of its mutual top-k neighbors
    (i in topk(j) and j in topk(i)); the track is pulled toward that target.
    No fixed similarity threshold, so discovery is active from the first
    epoch; mutual-neighbor filtering prevents pulling unrelated tracks.

    z: (U,D) L2-normalized.
    """
    U = z.shape[0]
    if U < 3:
        return torch.zeros((), device=z.device)
    sim = z @ z.t()
    sim = sim - torch.eye(U, device=z.device) * 1e3
    kk = min(k, U - 1)
    top = torch.topk(sim, kk, dim=1).indices  # (U,kk)
    has_target = torch.zeros(U, dtype=torch.bool, device=z.device)
    targets = torch.zeros_like(z)
    for i in range(U):
        nbrs = top[i]
        mutual = [int(j) for j in nbrs.tolist() if i in top[int(j)].tolist()]
        if not mutual:
            continue
        targets[i] = z[mutual].mean(dim=0)
        has_target[i] = True
    if not has_target.any():
        return torch.zeros((), device=z.device)
    t = l2norm_t(targets[has_target])
    zi = z[has_target]
    return (1.0 - (zi * t).sum(-1)).mean()


def open_space_loss(zu, anchors, open_thresh=0.60, open_margin=0.45):
    """Open-space repulsion: push self-selected likely-novel unlabeled
    trajectories away from the 48 known anchors.

    `likely-novel` is self-selected by max-known-sim < open_thresh (no labels,
    no benchmark novel vocabulary). The loss is a squared hinge above
    open_margin, so ambiguous unlabeled trajectories cannot be absorbed into
    known classes and are forced to form their own clusters (which MNN then
    sharpens).
    """
    sim = zu @ anchors.t()  # (U,K)
    maxsim = sim.max(dim=-1).values
    with torch.no_grad():
        mask = (maxsim < open_thresh).detach()
    if not mask.any():
        return torch.zeros((), device=zu.device)
    return torch.relu(maxsim[mask] - open_margin).pow(2).mean()


def tse_loss(
    model,
    anchors,
    known_feats,
    known_mask,
    known_labels,
    unlabeled_feats,
    unlabeled_mask,
    pca_class_means,
    tau=0.07,
    w_attr=1.0,
    w_frame=0.5,
    w_mnn=1.0,
    w_open=0.0,
    w_pres=0.1,
    mnn_k=8,
    open_thresh=0.60,
    open_margin=0.45,
):
    """known_feats: (B,T,D), unlabeled_feats: (U,T,D)."""
    zk, zkf = model.track_embed(known_feats, known_mask)
    zu, zuf = model.track_embed(unlabeled_feats, unlabeled_mask)
    an = anchors.normalized()
    logits = zk @ an.t() / tau
    ce = F.cross_entropy(logits, known_labels)
    attr = (1.0 - (zk * an[known_labels]).sum(-1)).mean()
    frame = (temporal_info_nce(zkf, zk, an, tau=tau)
             + temporal_info_nce(zuf, zu, an, tau=tau)) * 0.5
    mnn = mnn_attraction(zu, k=mnn_k)
    open_loss = open_space_loss(zu, an, open_thresh=open_thresh,
                                open_margin=open_margin)
    pres = anchor_preservation_loss(an, pca_class_means)
    # residual drift on known track means
    with torch.no_grad():
        raw_means = (known_feats * known_mask.unsqueeze(-1).float()).sum(1) / (
            known_mask.unsqueeze(-1).float().sum(1) + 1e-9)
        pca_target = model.pca_proj(raw_means)
    drift = (zk - pca_target).pow(2).mean()
    total = (ce + w_attr * attr + w_frame * frame + w_mnn * mnn
             + w_open * open_loss + w_pres * pres + w_pres * drift)
    return {
        "total": total,
        "ce": ce,
        "attr": attr,
        "frame": frame,
        "mnn": mnn,
        "open": open_loss,
        "pres": pres,
        "drift": drift,
    }
