"""Build trajectory-level class-conditional known explainability statistics.

Phase 7B uses a fixed (not learned) semantic explainability module:
per supported-known class, a diagonal multivariate Gaussian is estimated on
the corrected Phase 4T TRAIN stream (track-level causal EMA states). These
statistics are legal (supported-known labels only, no true novel GT) and are
used as evidence features inside the unified KNOWN/EXISTING/NEW competition.

The covariance is MAP-shrunk toward the pooled diagonal variance so that
classes with few tracks do not get degenerate likelihoods. A class with no
rows in the stream (anchor 60) falls back to its TSE anchor as the mean and
the pooled variance.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_episodes():
    tr = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/train_episodes.npz").items()}
    va = {k: np.asarray(v) for k, v in np.load(
        ROOT / "outputs/iclr27_phase7a/assets/val_episodes.npz").items()}
    ep = {}
    for k in tr:
        ep[k] = np.concatenate([tr[k], va[k]], axis=0)
    return ep


def replay_track_ema(ep, z_all, alpha=0.30):
    """Return per-row causal track-EMA states h (post-update)."""
    ema = {}
    h_all = np.zeros_like(z_all)
    for i in range(len(z_all)):
        key = (int(ep["video_ids"][i]), int(ep["track_ids"][i]))
        e = ema.get(key)
        z = z_all[i]
        if e is None:
            e = z.copy()
        else:
            e = (1 - alpha) * e + alpha * z
            e /= (np.linalg.norm(e) + 1e-12)
        ema[key] = e
        h_all[i] = e
    return h_all


def main():
    from src.iclr27_phase7a.training.train_reliability_head import load_tse, project

    dev = torch.device("cuda:0")
    model, anchors, known_ids = load_tse(dev)
    ep = load_episodes()
    z_all = project(dev, model, ep["feats"].astype(np.float32))
    h_all = replay_track_ema(ep, z_all)

    known_mask = ep["gt_role"] == 1
    cats = ep["gt_category_id"][known_mask].astype(np.int64)
    hs = h_all[known_mask]

    mu = {}
    var = {}
    cnt = {}
    for c in np.unique(cats):
        x = hs[cats == c]
        m = x.mean(axis=0)
        m /= (np.linalg.norm(m) + 1e-12)
        mu[int(c)] = m
        var[int(c)] = np.var(x, axis=0) + 1e-6
        cnt[int(c)] = len(x)

    pooled = np.zeros(128, dtype=np.float32)
    total = 0
    for c in mu:
        pooled = pooled + var[c] * cnt[c]
        total += cnt[c]
    pooled /= max(total, 1)

    # MAP shrinkage toward the pooled diagonal variance.
    lam = 0.5
    sigma2 = {}
    for c in mu:
        sigma2[c] = (1 - lam) * var[c] + lam * pooled
        sigma2[c] = np.clip(sigma2[c], 1e-4, 0.5)

    # Classes without rows: anchor as mean, pooled variance.
    anchor_idx = {int(k): i for i, k in enumerate(known_ids)}
    for c in known_ids.tolist():
        c = int(c)
        if c not in sigma2:
            mu[c] = anchors[anchor_idx[c]].astype(np.float32)
            sigma2[c] = pooled.copy()
            cnt[c] = 0

    order = [int(c) for c in known_ids]
    mus = np.stack([mu[c] for c in order]).astype(np.float32)
    sigs = np.stack([sigma2[c] for c in order]).astype(np.float32)
    logdet = np.log(sigs).sum(axis=1).astype(np.float32)
    counts = np.asarray([cnt[c] for c in order], dtype=np.int64)

    out = ROOT / "outputs/iclr27_phase7b/assets"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "known_stats.npz",
        known_ids=known_ids.astype(np.int64),
        mu=mus, sigma2=sigs, logdet=logdet, counts=counts,
        pooled=pooled, n_known_rows=int(known_mask.sum()),
        shrinkage=lam, ema_alpha=0.30,
    )
    stats = {
        "n_known_rows": int(known_mask.sum()),
        "n_classes": len(order),
        "classes_without_rows": sorted(
            int(c) for c in order if int(counts[int(np.where(known_ids == c)[0][0])]) == 0
        ),
        "mean_var": float(sigs.mean()),
        "min_var": float(sigs.min()),
        "max_var": float(sigs.max()),
    }
    (out / "known_stats_meta.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
