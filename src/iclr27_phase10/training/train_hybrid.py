"""Train a small causal hybrid representation prototype.

Only supported-known labels supervise the classifier.  Unlabeled TAO
trajectories contribute two-view temporal consistency; no novel category label
is loaded.  The expensive semantic memory/assign-create branch is absent.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase10.model.hybrid import HybridTrajectoryEncoder  # noqa: E402
from src.iclr27_phase7a.training.train_reliability_head import (  # noqa: E402
    load_tse,
    project,
)


def load_tracks():
    k = {n: np.asarray(v) for n, v in np.load(
        ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz").items()}
    u = {n: np.asarray(v) for n, v in np.load(
        ROOT / "outputs/iclr27_phase6c/assets/unlabeled_tracks.npz").items()}
    return k, u


def project_tracks(device, k, u):
    tse, _, _ = load_tse(device)
    def one(a):
        b, t, d = a["frame_feats"].shape
        z = project(device, tse, a["frame_feats"].reshape(b * t, d)
                    .astype(np.float32)).reshape(b, t, -1)
        return z.astype(np.float32)
    return one(k), one(u)


def augment(z, mask, rng, noise=0.015, drop=0.15):
    x = z.copy()
    m = mask.copy().astype(np.uint8)
    valid = np.argwhere(m > 0)
    if len(valid):
        keep = rng.rand(len(valid)) > drop
        # Keep the first valid observation so every view remains causal and
        # has a non-empty prefix.
        for (bi, ti), yes in zip(valid, keep):
            if ti == 0:
                continue
            if not yes:
                m[bi, ti] = 0
    x += rng.normal(0.0, noise, size=x.shape).astype(np.float32) * m[..., None]
    return x, m


def atomic_save_checkpoint(payload, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase10/training/hybrid_small")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps-per-epoch", type=int, default=40)
    ap.add_argument("--batch-known", type=int, default=64)
    ap.add_argument("--batch-unlabeled", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--w-consistency", type=float, default=1.0)
    ap.add_argument("--w-prefix", type=float, default=0.25)
    ap.add_argument("--w-known", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--no-consistency", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    device = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    k, u = load_tracks()
    kz, uz = project_tracks(device, k, u)
    known_ids = np.unique(k["labels"]).astype(np.int64)
    cat2idx = {int(c): i for i, c in enumerate(known_ids)}
    labels = np.asarray([cat2idx[int(c)] for c in k["labels"]], dtype=np.int64)
    model = HybridTrajectoryEncoder(dim=kz.shape[-1], hidden=args.hidden,
                                    out_dim=kz.shape[-1]).to(device)
    head = torch.nn.Linear(kz.shape[-1], len(known_ids)).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=args.weight_decay)

    logs_all = []
    for ep in range(args.epochs):
        t0 = time.time()
        accum = {"total": 0.0, "known": 0.0, "consistency": 0.0,
                 "prefix": 0.0}
        for _ in range(args.steps_per_epoch):
            ki = rng.choice(len(kz), size=min(args.batch_known, len(kz)),
                            replace=False)
            ui = rng.choice(len(uz), size=min(args.batch_unlabeled, len(uz)),
                            replace=False)
            kx = torch.from_numpy(kz[ki]).to(device)
            km = torch.from_numpy(k["frame_mask"][ki]).to(device)
            ux0, um0 = augment(uz[ui], u["frame_mask"][ui], rng)
            ux1, um1 = augment(uz[ui], u["frame_mask"][ui], rng)
            ux0, um0 = torch.from_numpy(ux0).to(device), torch.from_numpy(um0).to(device)
            ux1, um1 = torch.from_numpy(ux1).to(device), torch.from_numpy(um1).to(device)
            kh, kseq = model(kx, km)
            uh0, useq0 = model(ux0, um0)
            uh1, useq1 = model(ux1, um1)
            loss_known = F.cross_entropy(head(kh), torch.from_numpy(labels[ki]).to(device))
            loss_cons = (1.0 - F.cosine_similarity(uh0, uh1, dim=-1)).mean()
            if args.no_consistency:
                loss_cons = loss_cons.detach() * 0.0
            # Adjacent prefix agreement is computed within each causal view;
            # it is a temporal smoothness prior, not a semantic memory rule.
            loss_prefix = (1.0 - F.cosine_similarity(
                useq0[:, 1:], useq0[:, :-1], dim=-1)).mean()
            loss = args.w_known * loss_known + args.w_consistency * loss_cons \
                + args.w_prefix * loss_prefix
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            accum["total"] += float(loss.detach())
            accum["known"] += float(loss_known.detach())
            accum["consistency"] += float(loss_cons.detach())
            accum["prefix"] += float(loss_prefix.detach())
        for key in accum:
            accum[key] /= args.steps_per_epoch
        accum["epoch"] = ep + 1
        accum["seconds"] = time.time() - t0
        logs_all.append(accum)
        print(json.dumps(accum), flush=True)

    payload = {
        "model": {n: p.detach().cpu() for n, p in model.state_dict().items()},
        "head": {n: p.detach().cpu() for n, p in head.state_dict().items()},
        "known_ids": known_ids,
        "input": "frozen Phase-6C TSE frame projection",
        "tse_checkpoint": "outputs/iclr27_phase6c/training/tse_main/checkpoint.pth",
        "args": vars(args),
        "logs": logs_all,
    }
    atomic_save_checkpoint(payload, out / "checkpoint.pth")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    (out / "train_log.json").write_text(json.dumps(logs_all, indent=2))
    print("saved", out / "checkpoint.pth")


if __name__ == "__main__":
    main()
