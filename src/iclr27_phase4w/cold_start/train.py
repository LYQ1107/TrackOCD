"""Train ColdStartHead (3-way) or WarmMemoryHead (4-way).

Trained on meta-train category episodes; evaluated on category-disjoint
meta-dev episodes (TRAIN-only categories). Representations frozen; only
the decision head is trained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT


class ColdStartHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


class WarmMemoryHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 4))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["cold", "warm"], required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--meta-dev-samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(ROOT / args.samples)
    dv = np.load(ROOT / args.meta_dev_samples)
    Xtr = torch.from_numpy(d["cold_X" if args.head == "cold" else "warm_X"]).float()
    ytr = torch.from_numpy(d["cold_y" if args.head == "cold" else "warm_y"]).long()
    Xte = torch.from_numpy(dv["cold_X" if args.head == "cold" else "warm_X"]).float()
    yte = torch.from_numpy(dv["cold_y" if args.head == "cold" else "warm_y"]).long()
    dim = Xtr.shape[1]
    n_cls = 3 if args.head == "cold" else 4
    model = (ColdStartHead(dim) if args.head == "cold" else WarmMemoryHead(dim)).to(args.device)
    Xtr = Xtr.to(args.device); ytr = ytr.to(args.device)
    Xte = Xte.to(args.device); yte = yte.to(args.device)
    counts = torch.bincount(ytr, minlength=n_cls).float()
    weights = (ytr.numel() / (n_cls * counts.clamp(min=1))).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=args.device)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xtr[idx]), ytr[idx], weight=weights)
            loss.backward()
            opt.step()
            tot += float(loss)
        if (epoch + 1) % 15 == 0:
            print(f"ep {epoch+1} loss {tot/(len(perm)//args.batch_size+1):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        p_tr = torch.softmax(model(Xtr), -1)
        p_te = torch.softmax(model(Xte), -1)
    def report(p, y):
        yp = p.argmax(-1)
        acc = (yp == y).float().mean().item()
        rows = {}
        for c in range(n_cls):
            m = y == c
            if m.sum() == 0:
                continue
            pred = yp == c
            rows[str(c)] = {
                "recall": float((yp[m] == c).float().mean().item()),
                "precision": float((y[pred] == c).float().mean().item())
                if pred.any() else None,
                "n": int(m.sum()),
            }
        return acc, rows
    tr_acc, tr_rows = report(p_tr, ytr)
    te_acc, te_rows = report(p_te, yte)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "head": args.head, "dim": dim,
                "n_cls": n_cls, "args": vars(args)}, out / "head.pth")
    (out / "train_report.json").write_text(json.dumps({
        "head": args.head, "train_acc": tr_acc, "meta_dev_acc": te_acc,
        "train_rows": tr_rows, "meta_dev_rows": te_rows,
        "n_train": int(len(ytr)), "n_meta_dev": int(len(yte)),
    }, indent=2))
    print(json.dumps({
        "head": args.head, "train_acc": round(tr_acc, 4),
        "meta_dev_acc": round(te_acc, 4),
        "meta_dev_rows": te_rows,
    }, indent=2))


if __name__ == "__main__":
    main()
