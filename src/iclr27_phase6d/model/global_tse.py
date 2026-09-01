"""Phase 6D: TSE + Global Memory-Bank Neighborhood Aggregation (GMNA).

The student is the Phase 6C TSE (PCA-initialized base + residual MLP + 48
known anchors). A momentum teacher provides stable trajectory embeddings; a
global memory bank stores teacher embeddings for the full legal TAO TRAIN
pool; confidence-aware mutual-neighbor targets are mined in the bank and the
student is trained to match the aggregated target (teacher-student
consistency + neighborhood aggregation). Known CE / anchor preservation keep
known fidelity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase6c.model.tse import TSE, KnownAnchors


def l2norm_t(x, dim=-1):
    return F.normalize(x, dim=dim, eps=1e-12)


class GlobalTSE(nn.Module):
    def __init__(self, in_dim=768, hidden=256, out_dim=128,
                 teacher_momentum=0.999):
        super().__init__()
        self.student = TSE(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        self.teacher = TSE(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher_momentum = teacher_momentum

    def load_pca(self, pca_path: Path):
        self.student.load_pca(pca_path)
        self.teacher.load_pca(pca_path)
        self.teacher.load_state_dict(self.student.state_dict())

    @torch.no_grad()
    def update_teacher(self):
        m = self.teacher_momentum
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1 - m)
        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            if bt is not None and bs is not None:
                bt.data.copy_(bs.data)

    def project(self, x):
        return self.student.project(x)

    @torch.no_grad()
    def teacher_embed(self, x):
        return self.teacher.project(x)


class GlobalMemoryBank:
    """Stabilized bank of teacher trajectory embeddings + targets."""

    def __init__(self, n, dim, video_ids, sample_ids, alpha=0.90):
        self.n = n
        self.dim = dim
        self.register = False
        self.emb = torch.zeros(n, dim)
        self.video_ids = torch.as_tensor(video_ids, dtype=torch.int64)
        self.sample_ids = sample_ids
        self.alpha = alpha
        self.targets = torch.zeros(n, dim)
        self.has_target = torch.zeros(n, dtype=torch.bool)
        self.neighbor_counts = torch.zeros(n, dtype=torch.long)

    def to(self, device):
        self.emb = self.emb.to(device)
        self.video_ids = self.video_ids.to(device)
        self.targets = self.targets.to(device)
        self.has_target = self.has_target.to(device)
        self.neighbor_counts = self.neighbor_counts.to(device)
        return self

    @torch.no_grad()
    def update(self, new_emb):
        """new_emb: (n, dim) teacher embeddings for all tracks (L2 rows)."""
        if not self.register:
            self.emb.copy_(new_emb)
            self.register = True
        else:
            self.emb.mul_(self.alpha).add_(new_emb, alpha=1 - self.alpha)
        self.emb = l2norm_t(self.emb)

    @torch.no_grad()
    def build_targets(self, k=10, conf_min=0.45, prefer_cross_video=True,
                      anchors=None, novel_thresh=None):
        """Mutual top-k neighbors -> confidence-weighted mean target.

        If anchors + novel_thresh are given, only self-selected likely-novel
        tracks (teacher max-known-sim < novel_thresh) participate in the
        graph, preventing the known-dominated pool from absorbing novel
        trajectories (known-filtered bank).
        """
        E = self.emb
        sim = E @ E.t()
        same_video = (self.video_ids[:, None] == self.video_ids[None, :])
        same_track = torch.eye(self.n, dtype=torch.bool, device=E.device)
        mask = same_track | same_video
        if anchors is not None and novel_thresh is not None:
            max_k = (E @ anchors.t()).max(dim=-1).values
            novel_zone = max_k < novel_thresh
            mask = mask | (~novel_zone)[:, None] | (~novel_zone)[None, :]
        sim_masked = sim.clone()
        sim_masked[mask] = -1e9
        kk = min(k, self.n - 1)
        top = torch.topk(sim_masked, kk, dim=1).indices
        self.targets.zero_()
        self.has_target.zero_()
        self.neighbor_counts.zero_()
        for i in range(self.n):
            nbrs = top[i]
            # mutual with cross-video preference
            mutual = [int(j) for j in nbrs.tolist() if i in top[int(j)].tolist()]
            if not mutual and not prefer_cross_video:
                # fall back to non-mutual but confident cross-video neighbors
                mutual = [int(j) for j in nbrs.tolist()
                          if float(sim[i, j]) >= conf_min]
            if not mutual:
                continue
            w = torch.softmax(sim[i, mutual] / 0.07, dim=0)
            agg = (w[:, None] * E[mutual]).sum(dim=0)
            self.targets[i] = l2norm_t(agg)
            self.has_target[i] = True
            self.neighbor_counts[i] = len(mutual)

    def get_targets(self, idx):
        return self.targets[idx], self.has_target[idx]


def gmna_loss(student_z, teacher_z, targets, has_target, anchors,
              w_nb=2.0, w_ts=0.5, tau=0.07):
    """student_z / teacher_z: (B,D); targets: (B,D)."""
    loss_nb = torch.zeros((), device=student_z.device)
    if has_target.any():
        z = student_z[has_target]
        t = targets[has_target]
        loss_nb = (1.0 - (z * t).sum(-1)).mean()
    loss_ts = (1.0 - (student_z * teacher_z).sum(-1)).mean()
    return w_nb * loss_nb, w_ts * loss_ts, loss_nb, loss_ts
