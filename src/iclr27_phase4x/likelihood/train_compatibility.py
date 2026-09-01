"""X4: train learned trajectory-component compatibility (frozen states).

Positive pair: known state s_t vs anchor of its own category.
Negative pair: state vs other known anchors, and pseudo-novel states vs
every active known anchor. NEW/NOISE remain null hypotheses in inference.
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)


def load_tsr(device):
    from src.iclr27_phase4u.trajectory.model import TSR
    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=device)
    sd = {k[len("rep."):]: v for k, v in ck["model"].items() if k.startswith("rep.")}
    tsr = TSR(arch="gru").to(device)
    tsr.load_state_dict(sd)
    tsr.eval()
    return tsr


class CompatibilityNet(nn.Module):
    def __init__(self, dim: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, s: torch.Tensor, mu: torch.Tensor):
        sn = F.normalize(s, dim=-1)
        mn = F.normalize(mu, dim=-1)
        if sn.shape[0] == 1 and mn.shape[0] > 1:
            sn = sn.expand(mn.shape[0], -1)
        x = torch.cat([sn, mn, sn - mn], dim=-1)
        return self.net(x).squeeze(-1)


def build_pairs(n_episodes, seed, device, store=None, max_steps=3):
    store = store or load_store()
    cfg = WEpisodeConfig()
    tsr = load_tsr(device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"]
    rng = random.Random(seed)
    X, Y = [], []
    with torch.no_grad():
        for e in range(n_episodes):
            ep = make_episode(store, pool, cfg, rng)
            active_idx = [cat_index[c] for c in ep["pseudo_known"]]
            active_cats = ep["pseudo_known"]
            for occ in ep["occurrences"]:
                z, q = store.tracklet_seq(occ["key"])
                n = min(len(z), cfg.max_len)
                state = tsr.init_state(1, device)
                steps = list(range(cfg.min_commit_age - 1, n))
                rng.shuffle(steps)
                steps = steps[:max_steps]
                for t in steps:
                    f = torch.from_numpy(z[t:t + 1]).to(device)
                    qt = torch.from_numpy(q[t:t + 1]).to(device)
                    s, state = tsr.step(f, qt, state)
                    if occ["role"] == "known":
                        c = occ["category"]
                        if c in active_cats:
                            mu = anchors[cat_index[c]]
                            X.append(torch.cat([s[0], mu]).cpu().numpy())
                            Y.append(1)
                            negs = [ci for ci in active_idx if cat_ids[ci] != c]
                            rng.shuffle(negs)
                            for ci in negs[:3]:
                                mu = anchors[ci]
                                X.append(torch.cat([s[0], mu]).cpu().numpy())
                                Y.append(0)
                    elif occ["role"] == "novel":
                        for ci in active_idx:
                            mu = anchors[ci]
                            X.append(torch.cat([s[0], mu]).cpu().numpy())
                            Y.append(0)
            if (e + 1) % 100 == 0:
                print(f"episode {e+1}/{n_episodes}", flush=True)
    return np.stack(X).astype(np.float32), np.asarray(Y, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X, Y = build_pairs(args.n_episodes, args.seed, args.device)
    Xt = torch.from_numpy(X).to(args.device)
    Yt = torch.from_numpy(Y).to(args.device)
    model = CompatibilityNet().to(args.device)
    ymean = Yt.float().mean()
    w = torch.tensor([ymean, 1 - ymean], device=args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=args.device)
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            s = Xt[idx, :256]
            mu = Xt[idx, 256:]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(s, mu), Yt[idx].float(), weight=w[Yt[idx]])
            loss.backward()
            opt.step()
            tot += float(loss)
        if (epoch + 1) % 10 == 0:
            print(f"ep {epoch+1} loss {tot:.4f}", flush=True)
    # eval: pos vs neg score separation
    model.eval()
    with torch.no_grad():
        pos = model(Xt[Yt == 1, :256], Xt[Yt == 1, 256:]).cpu().numpy()
        neg = model(Xt[Yt == 0, :256], Xt[Yt == 0, 256:]).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]),
                              np.concatenate([pos, neg])))
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, out / "compat.pth")
    (out / "report.json").write_text(json.dumps({
        "n_pairs": int(len(Y)), "pos_rate": round(float(Y.mean()), 4),
        "pos_mean": round(float(pos.mean()), 4),
        "neg_mean": round(float(neg.mean()), 4),
        "auroc": round(auc, 4),
    }, indent=2))
    print(json.dumps({"n_pairs": int(len(Y)), "auroc": round(auc, 4),
                      "pos_mean": round(float(pos.mean()), 4),
                      "neg_mean": round(float(neg.mean()), 4)}, indent=2))


if __name__ == "__main__":
    main()
