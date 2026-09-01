"""Train Phase 4Z routing candidates on genuine-OOV episodes.

Candidates:
  gru         : causal GRU evidence accumulator over [h_t; ev_t] (proposed)
  static      : static prefix embedding MLP over [h_T; ev_T] (reviewer control)
  meanpool    : causal mean-pooled prefix MLP over mean([h; ev]) (control)
  aggregated  : trajectory-aggregated non-sequential MLP (control)
  singleframe : first-frame-only MLP over [h_1; ev_1] (control)

Supervision: decision-age-sampled labels. For each known/novel sequence a
decision age D is sampled per epoch (immediate decisions are common, later
decisions teach evidence accumulation); steps before D are UNRESOLVED, step
D-1 carries the true role, later steps are masked. FP sequences are always
UNRESOLVED. Model selection uses category-disjoint meta-dev only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT


class GRURouter(nn.Module):
    def __init__(self, in_dim: int = 275, hidden: int = 96):
        super().__init__()
        self.gru = nn.GRUCell(in_dim, hidden)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))

    def forward_seq(self, xs):
        """xs: (T, B, in_dim) -> (T, B, 3)."""
        h = torch.zeros(xs.shape[1], self.gru.hidden_size, device=xs.device)
        out = []
        for t in range(xs.shape[0]):
            h = self.gru(xs[t], h)
            out.append(self.head(self.drop(h)))
        return torch.stack(out, dim=0)


class MLPRouter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def load_episodes(path: Path):
    d = np.load(path)
    ev = d["ev"]
    h = d["h"]
    y_role = d["y_role"]
    starts = d["seq_start"]
    roles = d["seq_role"]
    lens = np.diff(starts)
    seqs = []
    for i, (s, ln) in enumerate(zip(starts[:-1], lens)):
        seqs.append({
            "ev": ev[s:s + ln].astype(np.float32),
            "h": h[s:s + ln].astype(np.float32),
            "role": int(roles[i]),
            "len": int(ln),
        })
    return seqs


def sample_decision_age(role: int, length: int, rng: random.Random,
                        d1_prob: float = 0.35):
    if role == 2 or length <= 1:
        return None
    r = rng.random()
    if r < d1_prob:
        d = 1
    elif r < d1_prob + 0.25:
        d = 2
    elif r < d1_prob + 0.40:
        d = 3
    elif r < d1_prob + 0.50:
        d = 4
    elif r < d1_prob + 0.60:
        d = 5 if length >= 5 else length
    else:
        d = rng.randint(6, max(6, length))
    return min(d, length)


def feature_sets(seqs, mode):
    """Per-step feature vectors for non-recurrent modes."""
    out = []
    for sq in seqs:
        ev, h = sq["ev"], sq["h"]
        T = len(ev)
        if mode == "static":
            for t in range(T):
                out.append((np.concatenate([h[t], ev[t]]), t))
        elif mode == "meanpool":
            for t in range(T):
                out.append((np.concatenate([h[:t + 1].mean(0), ev[:t + 1].mean(0)]), t))
        elif mode == "singleframe":
            out.append((np.concatenate([h[0], ev[0]]), 0))
        elif mode == "aggregated":
            for t in range(T):
                ee = ev[:t + 1]
                agg = np.concatenate([
                    ee.mean(0), ee.max(0), ee.min(0), ee.std(0),
                    ev[t], h[:t + 1].mean(0), h[t],
                    np.array([min(t + 1, 16) / 16.0], np.float32),
                ]).astype(np.float32)
                out.append((agg, t))
    return out


def make_labels(seqs, seed, scheme="sampled", d1_prob=0.35):
    rng = random.Random(seed)
    labels = []
    for sq in seqs:
        T = sq["len"]
        role = sq["role"]
        D = sample_decision_age(role, T, rng, d1_prob) if scheme == "sampled" else (
            None if role == 2 else 1)
        y = np.full(T, 2, dtype=np.int64)
        mask = np.ones(T, dtype=bool)
        if D is not None:
            if scheme == "sampled":
                mask[D:] = False
                y[D - 1] = role
            else:
                y[:] = role
        labels.append((y, mask, D))
    return labels


def evaluate_online(model, seqs, mode, device, tau):
    model.eval()
    res = []
    with torch.no_grad():
        for sq in seqs:
            ev, h = sq["ev"], sq["h"]
            T = len(ev)
            commit = None
            if mode == "gru":
                xs = torch.from_numpy(np.concatenate([h, ev], axis=1)).unsqueeze(1).to(device)
                logits = model.forward_seq(xs)[:, 0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            else:
                feats = feature_sets([sq], mode)
                probs = []
                for f, t in feats:
                    x = torch.from_numpy(f).unsqueeze(0).to(device)
                    probs.append(torch.softmax(model(x), dim=-1).cpu().numpy()[0])
                probs = np.stack(probs)
            for t in range(len(probs)):
                p = probs[t]
                a = int(p.argmax())
                if a != 2 and p[a] >= tau:
                    commit = (a, t + 1)
                    break
            res.append({"role": sq["role"], "commit": commit,
                        "len": T, "probs": probs})
    return res


def routing_metrics(res):
    n_known = sum(1 for r in res if r["role"] == 0)
    n_novel = sum(1 for r in res if r["role"] == 1)
    n_fp = sum(1 for r in res if r["role"] == 2)
    known_rr = sum(1 for r in res if r["role"] == 0 and r["commit"] and r["commit"][0] == 0) / max(n_known, 1)
    novel_rr = sum(1 for r in res if r["role"] == 1 and r["commit"] and r["commit"][0] == 1) / max(n_novel, 1)
    known_to_novel = sum(1 for r in res if r["role"] == 0 and r["commit"] and r["commit"][0] == 1)
    novel_to_known = sum(1 for r in res if r["role"] == 1 and r["commit"] and r["commit"][0] == 0)
    un_known = sum(1 for r in res if r["role"] == 0 and (not r["commit"] or r["commit"][0] == 2))
    un_novel = sum(1 for r in res if r["role"] == 1 and (not r["commit"] or r["commit"][0] == 2))
    fp_committed = sum(1 for r in res if r["role"] == 2 and r["commit"])
    return {
        "n": len(res), "n_known": n_known, "n_novel": n_novel, "n_fp": n_fp,
        "known_rr": float(known_rr), "novel_rr": float(novel_rr),
        "known_to_novel": known_to_novel, "novel_to_known": novel_to_known,
        "unresolved_known": un_known, "unresolved_novel": un_novel,
        "fp_committed": fp_committed,
        "balanced": float((known_rr + novel_rr) / 2),
        "mean_commit_age": float(np.mean([r["commit"][1] for r in res if r["commit"]])) if any(r["commit"] for r in res) else None,
    }


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    train_seqs = load_episodes(ROOT / args.train_episodes)
    dev_seqs = load_episodes(ROOT / args.meta_dev_episodes)
    print("train seqs", len(train_seqs), "meta-dev seqs", len(dev_seqs), flush=True)
    ev_dim = train_seqs[0]["ev"].shape[1]

    if args.normalize:
        all_ev = np.concatenate([sq["ev"] for sq in train_seqs], axis=0)
        all_h = np.concatenate([sq["h"] for sq in train_seqs], axis=0)
        ev_mean = all_ev.mean(0).astype(np.float32)
        ev_std = all_ev.std(0).astype(np.float32) + 1e-6
        h_mean = all_h.mean(0).astype(np.float32)
        h_std = all_h.std(0).astype(np.float32) + 1e-6
        for sq in train_seqs + dev_seqs:
            sq["ev"] = (sq["ev"] - ev_mean) / ev_std
            sq["h"] = (sq["h"] - h_mean) / h_std
    else:
        ev_mean = np.zeros(ev_dim, np.float32)
        ev_std = np.ones(ev_dim, np.float32)
        h_mean = np.zeros(256, np.float32)
        h_std = np.ones(256, np.float32)

    if args.mode == "gru":
        model = GRURouter(256 + ev_dim, args.hidden).to(device)
        params = list(model.parameters())
    else:
        if args.mode in ("static", "meanpool"):
            in_dim = 256 + ev_dim
        elif args.mode == "singleframe":
            in_dim = 256 + ev_dim
        elif args.mode == "aggregated":
            in_dim = ev_dim * 4 + ev_dim + 256 + 256 + 1
        model = MLPRouter(in_dim, args.hidden).to(device)
        params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=1e-3 if args.mode == "gru" else 1e-4)
    # Defer is cheap before the sampled decision age; committing correctly at
    # the decision age is the primary objective. FP sequences remain defer.
    weights = torch.tensor([1.0, 1.0, args.defer_weight], device=device)

    n_params = sum(p.numel() for p in params)
    print("mode", args.mode, "params", n_params, flush=True)
    best_bal = -1.0
    best_sd = None

    for epoch in range(args.epochs):
        model.train()
        labels = make_labels(train_seqs, args.seed * 1000 + epoch,
                             args.label_scheme, args.d1_prob)
        total = 0.0
        n_steps = 0
        if args.mode == "gru":
            order = list(range(len(train_seqs)))
            random.Random(args.seed + epoch).shuffle(order)
            for bi in range(0, len(order), args.seq_batch):
                idx = order[bi:bi + args.seq_batch]
                T = max(train_seqs[i]["len"] for i in idx)
                ev = np.zeros((len(idx), T, ev_dim), np.float32)
                h = np.zeros((len(idx), T, 256), np.float32)
                yb = np.full((len(idx), T), 2, dtype=np.int64)
                maskb = np.zeros((len(idx), T), dtype=bool)
                for j, i in enumerate(idx):
                    sq = train_seqs[i]
                    n = sq["len"]
                    ev[j, :n] = sq["ev"]
                    h[j, :n] = sq["h"]
                    yb[j, :n] = labels[i][0]
                    maskb[j, :n] = labels[i][1]
                xs = torch.from_numpy(np.concatenate([h, ev], axis=2)).permute(1, 0, 2).to(device)
                logits = model.forward_seq(xs)
                yt = torch.from_numpy(yb).permute(1, 0).to(device)
                loss = F.cross_entropy(
                    logits[maskb.T], yt[maskb.T],
                    weight=weights)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * int(maskb.sum())
                n_steps += int(maskb.sum())
        else:
            feats_all = feature_sets(train_seqs, args.mode)
            # align feature -> sequence index
            seq_ids = []
            for i, sq in enumerate(train_seqs):
                T = sq["len"]
                n_feat = 1 if args.mode == "singleframe" else T
                seq_ids.extend([i] * n_feat)
            order = list(range(len(feats_all)))
            random.Random(args.seed + epoch).shuffle(order)
            opt.zero_grad()
            for bi in range(0, len(order), args.batch_size):
                idx = order[bi:bi + args.batch_size]
                X = np.stack([feats_all[i][0] for i in idx])
                yb = np.zeros(len(idx), dtype=np.int64)
                maskb = np.zeros(len(idx), dtype=bool)
                for j, i in enumerate(idx):
                    si = seq_ids[i]
                    t = feats_all[i][1]
                    yb[j], maskb[j], _ = labels[si][0][t], labels[si][1][t], None
                x = torch.from_numpy(X).to(device)
                yt = torch.from_numpy(yb[maskb]).to(device)
                if yt.numel() == 0:
                    continue
                logits = model(x[maskb])
                loss = F.cross_entropy(logits, yt, weight=weights)
                loss.backward()
                if (bi // args.batch_size + 1) % args.accum == 0:
                    opt.step()
                    opt.zero_grad()
                total += float(loss) * int(maskb.sum())
                n_steps += int(maskb.sum())
            opt.step()
            opt.zero_grad()
        if (epoch + 1) % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch+1} loss {total/max(n_steps,1):.4f}", flush=True)
        if args.mode == "gru" and ((epoch + 1) % 4 == 0 or epoch + 1 == args.epochs):
            res = evaluate_online(model, dev_seqs, args.mode, device, 0.3)
            m = routing_metrics(res)
            print(f"  meta-dev epoch {epoch+1} balanced {m['balanced']:.4f} "
                  f"(krr {m['known_rr']:.3f} nrr {m['novel_rr']:.3f})", flush=True)
            if m["balanced"] > best_bal:
                best_bal = m["balanced"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)
        print("restored best meta-dev epoch, balanced", round(best_bal, 4), flush=True)

    # meta-dev threshold sweep
    best = None
    rows = []
    for tau in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        res = evaluate_online(model, dev_seqs, args.mode, device, tau)
        m = routing_metrics(res)
        m["tau"] = tau
        rows.append(m)
        if best is None or m["balanced"] > best["balanced"]:
            best = m
    print("meta-dev best:", json.dumps(best, indent=2), flush=True)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mode": args.mode,
                "args": vars(args), "n_params": n_params,
                "ev_mean": ev_mean, "ev_std": ev_std,
                "h_mean": h_mean, "h_std": h_std}, out / "router.pth")
    (out / "meta_dev.json").write_text(json.dumps({
        "best": best, "sweep": rows,
    }, indent=2))
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["gru", "static", "meanpool", "aggregated", "singleframe"])
    ap.add_argument("--train-episodes", required=True)
    ap.add_argument("--meta-dev-episodes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--defer-weight", type=float, default=0.25)
    ap.add_argument("--normalize", type=int, default=1)
    ap.add_argument("--seq-batch", type=int, default=64)
    ap.add_argument("--label-scheme", choices=["sampled", "flat"], default="sampled")
    ap.add_argument("--d1-prob", type=float, default=0.35)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
