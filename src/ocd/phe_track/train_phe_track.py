#!/usr/bin/env python3
"""Train PHE-Track on known-class track embeddings using the official PHE
training protocol (200 epochs, AdamW, cosine LR, CPG+DCE losses)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ocd.phe_track.phe_track_model import (
    PPNetTrack,
    cos_eps_loss,
    get_dis_max,
    sep_loss,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(encoder):
    cache = PROJECT_ROOT / "data" / "caches" / "features" / encoder / "train_known_mean"
    feats = {}
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    labels = {}
    with open(PROJECT_ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] in feats:
                labels[r["sample_id"]] = r["category_id"]
    class_ids = sorted(set(labels.values()))
    cid2idx = {c: i for i, c in enumerate(class_ids)}
    X = np.stack([feats[s] for s in labels])
    y = np.array([cid2idx[labels[s]] for s in labels])
    return X, y, class_ids


def train_one_epoch(model, ema_model, X, y, optimizer, scheduler, epoch, args, device):
    model.train()
    order = torch.randperm(len(X))
    losses = []
    dis_max = get_dis_max(args.hash_code_length, len(set(y.tolist())))
    for i in range(0, len(order), args.batch_size):
        idx = order[i : i + args.batch_size]
        xb = torch.from_numpy(X[idx]).to(device)
        yb = torch.from_numpy(y[idx]).to(device)
        optimizer.zero_grad()
        logits, hash_feat = model(xb)
        loss_protop = F.nll_loss(F.log_softmax(logits, dim=1), yb)

        class_means = torch.stack(
            [
                model.prototype_vectors_global[c : c + model.global_proto_per_class].mean(0)
                for c in range(0, model.num_prototypes_global, model.global_proto_per_class)
            ]
        )
        hash_centers = model.hash_head(class_means)
        hash_centers_sign = torch.tanh(hash_centers * 3)
        loss_sep = sep_loss(hash_centers_sign, L=args.hash_code_length, dis_max=dis_max)
        loss_quan = (1 - torch.abs(hash_centers_sign)).mean()
        loss_feat = cos_eps_loss(hash_feat, yb, hash_centers)
        loss = loss_protop + loss_sep * args.alpha + loss_quan * args.alpha + loss_feat * args.beta
        loss.backward()
        optimizer.step()
        if ema_model is not None:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    ema_p.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)
        losses.append(loss.item())
    if scheduler is not None:
        scheduler.step()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["dinov2", "clip"], required=True)
    ap.add_argument("--seed", type=int, default=1027)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hash_code_length", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ema_decay", type=float, default=0.99996)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    X, y, class_ids = load_data(args.encoder)
    K = len(class_ids)
    print(f"train samples={len(X)} classes={K}", flush=True)

    model = PPNetTrack(
        in_dim=X.shape[1],
        prototype_dim=X.shape[1],
        num_classes=K,
        global_proto_per_class=10,
        hash_code_length=args.hash_code_length,
    ).to(device)
    ema = PPNetTrack(
        in_dim=X.shape[1],
        prototype_dim=X.shape[1],
        num_classes=K,
        global_proto_per_class=10,
        hash_code_length=args.hash_code_length,
    ).to(device)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": args.lr},
            {"params": model.add_on_layers.parameters(), "lr": args.lr * 10},
            {"params": model.prototype_vectors_global, "lr": args.lr * 10},
            {"params": model.hash_head.parameters(), "lr": args.lr * 10},
        ],
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = PROJECT_ROOT / "runs" / "phe_track" / f"{args.encoder}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for epoch in range(args.epochs):
        loss = train_one_epoch(
            model, ema, X, y, optimizer, scheduler, epoch, args, device
        )
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"epoch {epoch+1}/{args.epochs} loss={loss:.4f}", flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "class_ids": class_ids,
            "args": vars(args),
            "seed": args.seed,
            "epochs": args.epochs,
        },
        out_dir / "checkpoint.pth",
    )
    print(f"training done in {time.time()-t0:.1f}s -> {out_dir / 'checkpoint.pth'}", flush=True)


if __name__ == "__main__":
    main()
