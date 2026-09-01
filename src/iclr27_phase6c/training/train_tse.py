"""Train the Phase 6C Trajectory Semantic Encoder.

Data: outputs/iclr27_phase6c/assets/{known,unlabeled}_tracks.npz + pca.npz.
Protocol: known CE + anchor attraction + same-track temporal InfoNCE +
cross-track MNN attraction + anchor-preservation drift; class-balanced
known sampling; no novel GT used.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
from src.iclr27_phase6c.model.tse import TSE, KnownAnchors, tse_loss


def load_assets():
    k = {name: np.asarray(arr) for name, arr in np.load(
        ROOT / "outputs/iclr27_phase6c/assets/known_tracks.npz").items()}
    u = {name: np.asarray(arr) for name, arr in np.load(
        ROOT / "outputs/iclr27_phase6c/assets/unlabeled_tracks.npz").items()}
    ug = {name: np.asarray(arr) for name, arr in np.load(
        ROOT / "outputs/iclr27_phase6c/assets/unlabeled_gt_tracks.npz").items()}
    return k, u, ug


def pca_class_means(k):
    pca = np.load(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    comp = pca["components"].astype(np.float32)  # (128,768)
    mean = pca["mean"].astype(np.float32)
    labels = k["labels"]
    known_ids = np.unique(labels)
    out = []
    for c in known_ids:
        idx = np.where(labels == c)[0]
        m = k["mean_feats"][idx].mean(axis=0)
        v = (m - mean) @ comp.T
        v = v / (np.linalg.norm(v) + 1e-12)
        out.append(v)
    return known_ids, np.stack(out)


def sample_known(k, cat2idx, n_per_class, rng, max_frames=8):
    labels = k["labels"]
    classes = np.unique(labels)
    pick = []
    for c in classes:
        idx = np.where(labels == c)[0]
        cnt = rng.randint(1, n_per_class + 1) if rng.random() < 0.75 else 0
        if cnt:
            pick.extend(rng.choice(idx, size=cnt, replace=True).tolist())
    if not pick:
        pick = rng.choice(len(labels), size=32, replace=True).tolist()
    pick = np.asarray(pick)
    idx_labels = np.array([cat2idx[int(c)] for c in labels[pick]], dtype=np.int64)
    return (
        torch.from_numpy(k["frame_feats"][pick].astype(np.float32)),
        torch.from_numpy(k["frame_mask"][pick]),
        torch.from_numpy(idx_labels),
    )


def sample_unlabeled(u, batch, rng, max_frames=8):
    idx = rng.choice(len(u["mean_feats"]), size=batch, replace=False)
    return (
        torch.from_numpy(u["frame_feats"][idx].astype(np.float32)),
        torch.from_numpy(u["frame_mask"][idx]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase6c/training/tse_main")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--known-per-class", type=int, default=2)
    ap.add_argument("--unlabeled-batch", type=int, default=32)
    ap.add_argument("--unlabeled-gt-batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--w-mnn", type=float, default=1.0)
    ap.add_argument("--w-frame", type=float, default=0.5)
    ap.add_argument("--w-pres", type=float, default=0.1)
    ap.add_argument("--w-attr", type=float, default=1.0)
    ap.add_argument("--w-open", type=float, default=0.0)
    ap.add_argument("--open-thresh", type=float, default=0.60)
    ap.add_argument("--open-margin", type=float, default=0.45)
    ap.add_argument("--mnn-k", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = torch.device(args.device)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    k, u, ug = load_assets()
    known_ids, pca_means = pca_class_means(k)
    cat2idx = {int(c): i for i, c in enumerate(known_ids)}
    model = TSE().to(dev)
    model.load_pca(ROOT / "outputs/iclr27_phase6c/assets/pca.npz")
    anchors = KnownAnchors(known_ids, init_feats=pca_means).to(dev)
    params = [{"params": model.parameters()},
              {"params": anchors.parameters(), "lr": args.lr * 0.5}]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, int(20000 / max(48 + args.unlabeled_batch, 1)))
    if args.max_steps:
        steps_per_epoch = args.max_steps
    sched = CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    n_known = int(np.unique(k["labels"]).shape[0] * args.known_per_class * 0.75)
    print(f"known tracks {len(k['labels'])}, unlabeled {len(u['mean_feats'])}, "
          f"unlabeled_gt {len(ug['mean_feats'])}, "
          f"steps/epoch {steps_per_epoch}, classes {len(known_ids)}")
    for ep in range(args.epochs):
        t0 = time.time()
        logs = {}
        for step in range(steps_per_epoch):
            kf, km, kl = sample_known(k, cat2idx, args.known_per_class, rng)
            uf_p, um_p = sample_unlabeled(u, args.unlabeled_batch, rng)
            uf_g, um_g = sample_unlabeled(ug, args.unlabeled_gt_batch, rng)
            uf = torch.cat([uf_p, uf_g], dim=0)
            um = torch.cat([um_p, um_g], dim=0)
            kf, km, kl = kf.to(dev), km.to(dev), kl.to(dev)
            uf, um = uf.to(dev), um.to(dev)
            lossd = tse_loss(
                model, anchors, kf, km, kl, uf, um,
                pca_means,
                tau=0.07, w_attr=args.w_attr, w_frame=args.w_frame,
                w_mnn=args.w_mnn, w_pres=args.w_pres, w_open=args.w_open,
                open_thresh=args.open_thresh, open_margin=args.open_margin,
                mnn_k=args.mnn_k,
            )
            opt.zero_grad()
            lossd["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(anchors.parameters()), 5.0)
            opt.step()
            sched.step()
            for name, v in lossd.items():
                logs[name] = logs.get(name, 0.0) + float(v.detach())
        for name in logs:
            logs[name] /= steps_per_epoch
        print(f"epoch {ep + 1}/{args.epochs} "
              f"{time.time() - t0:.1f}s " +
              " ".join(f"{name}={v:.4f}" for name, v in logs.items()), flush=True)
        if (ep + 1) % args.save_every == 0 or (ep + 1) == args.epochs:
            torch.save({
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "anchors": {k: v.detach().cpu() for k, v in anchors.state_dict().items()},
                "known_ids": known_ids,
                "epoch": ep + 1,
            }, out / f"checkpoint_{ep + 1:03d}.pth")
    torch.save({
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "anchors": {k: v.detach().cpu() for k, v in anchors.state_dict().items()},
        "known_ids": known_ids,
        "epoch": args.epochs,
    }, out / "checkpoint.pth")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    print("saved", out / "checkpoint.pth")


if __name__ == "__main__":
    main()
