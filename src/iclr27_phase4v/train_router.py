"""Train the independent open-world router (logistic or small MLP).

Labels: known=1, novel=0 (fp excluded from the 2-way pilot). Router consumes
the 15-dim dual-space evidence vector. Held-out evaluation is by episode
(last 20% of episodes), so the router is tested on pseudo-novel categories
not seen during training.
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


class LogisticRouter(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Linear(dim, 2)

    def forward(self, x):
        return self.net(x)


class MLPRouter(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def auroc_auprc(score: np.ndarray, y: np.ndarray):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return (float(roc_auc_score(y, score)),
            float(average_precision_score(y, score)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="mlp", choices=["logistic", "mlp"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    d = np.load(ROOT / args.samples)
    X, y, ep = d["X"], d["y"], d["ep_idx"]
    mask = y <= 1
    X, y, ep = X[mask], y[mask], ep[mask]
    # held-out by episode: episodes >= 0.8*n_ep
    n_ep = int(ep.max()) + 1
    te_mask = ep >= int(0.8 * n_ep)
    Xtr, ytr = X[~te_mask], y[~te_mask]
    Xte, yte = X[te_mask], y[te_mask]
    print("train", Xtr.shape, "test", Xte.shape,
          "pos rate", ytr.mean().round(3), yte.mean().round(3))

    model = (LogisticRouter(X.shape[1]) if args.arch == "logistic"
             else MLPRouter(X.shape[1])).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    Xt = torch.from_numpy(Xtr).to(args.device)
    yt = torch.from_numpy(ytr).to(args.device)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=args.device)
        loss_sum = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            loss_sum += float(loss)
        if (epoch + 1) % 10 == 0:
            print(f"ep {epoch+1} loss {loss_sum:.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        p_tr = F.softmax(model(torch.from_numpy(Xtr).to(args.device)), -1)[:, 1].cpu().numpy()
        p_te = F.softmax(model(torch.from_numpy(Xte).to(args.device)), -1)[:, 1].cpu().numpy()
    tr_auc, tr_apr = auroc_auprc(p_tr, ytr)
    te_auc, te_apr = auroc_auprc(p_te, yte)
    report = {
        "arch": args.arch, "dim": int(X.shape[1]),
        "train_auroc": tr_auc, "train_auprc": tr_apr,
        "heldout_auroc": te_auc, "heldout_auprc": te_apr,
        "n_train": int(len(ytr)), "n_heldout": int(len(yte)),
        "n_episodes": n_ep,
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": args.arch,
                "dim": int(X.shape[1]),
                "args": vars(args)}, out / "router.pth")
    (out / "router_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
