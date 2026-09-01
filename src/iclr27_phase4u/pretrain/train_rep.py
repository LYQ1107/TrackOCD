"""Representation-only pretraining: cross-physical-track semantic contrast.

Stage A of Phase 4U. Loss:
  L = lambda_ct * SupCon(same category, different physical instance)
    + lambda_ce * prototype CE
    + lambda_temp * temporal consistency (same-instance prefixes)

Supervision variants (ablation A): cross_track (default), same_physical
(ReID-style: only same-instance positives), none (CE only).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT, load_source
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4u.data import class_sets


def supcon_loss(s: torch.Tensor, cats: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    N = s.shape[0]
    sim = s @ s.t() / temp
    eye = torch.eye(N, dtype=torch.bool, device=s.device)
    pos_mask = (cats[:, None] == cats[None, :]) & (~eye)
    if not pos_mask.any():
        return torch.zeros((), device=s.device)
    exp = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
    exp = exp * (~eye)
    denom = exp.sum(dim=1) + 1e-9
    log_probs = torch.log(exp / denom[:, None] + 1e-12)
    losses = []
    for i in range(N):
        pos = pos_mask[i]
        if pos.any():
            losses.append(-log_probs[i][pos].mean())
    return torch.stack(losses).mean()


class PairSampler:
    def __init__(self, sources: list[str], seed: int):
        self.sources = [load_source(s) for s in sources]
        self.insts = []
        for src in self.sources:
            for x in src["instances"]:
                self.insts.append({
                    "id": x["id"], "cat": x["cat"], "feats": x["feats"],
                    "q": x["q"], "src": src["name"],
                })
        self.by_cat: dict[int, list[dict]] = defaultdict(list)
        for x in self.insts:
            self.by_cat[x["cat"]].append(x)
        self.rng = random.Random(seed)

    def sample_batch(self, n_classes: int, k: int, max_len: int):
        cats_with_pos = [c for c, v in self.by_cat.items() if len(v) >= 2]
        self.rng.shuffle(cats_with_pos)
        chosen = cats_with_pos[:n_classes]
        batch = []
        for c in chosen:
            pool = self.by_cat[c][:]
            self.rng.shuffle(pool)
            batch.extend(pool[:k])
        self.rng.shuffle(batch)
        feats = []
        qs = []
        cats = []
        temp_pairs = []  # (i, len1, len2)
        lens1 = []
        lens2 = []
        for i, x in enumerate(batch):
            T = x["feats"].shape[0]
            l1 = min(T, self.rng.choice([1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8, 10, 12, 16]))
            lens1.append(l1)
            cats.append(x["cat"])
            if T >= 2:
                l2 = min(T, l1 + self.rng.choice([1, 2, 3]))
                temp_pairs.append((i, l1, l2))
                lens2.append(l2)
            else:
                l2 = l1
                lens2.append(l1)
            lmax = max(l1, l2)
            feats.append(x["feats"][:lmax])
            qs.append(x["q"][:lmax] if x["q"] is not None else None)
        return batch, feats, qs, cats, temp_pairs, lens1, lens2


def build_padded_batch(feats: list, qs: list, lens: list[int], device):
    """Pad variable-length prefixes into (B,T,768), (B,T,6), (B,T) bool."""
    B = len(feats)
    T = max(lens)
    f = torch.zeros(B, T, 768, device=device)
    q = torch.zeros(B, T, 6, device=device)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i in range(B):
        n = lens[i]
        f[i, :n] = torch.from_numpy(feats[i][:n]).to(device)
        if qs[i] is not None:
            q[i, :n] = torch.from_numpy(qs[i][:n]).to(device)
        mask[i, :n] = True
    return f, q, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="mixed", choices=["real", "episodic", "mixed"])
    ap.add_argument("--arch", default="gru", choices=["gru", "mean"])
    ap.add_argument("--supervision", default="cross_track",
                    choices=["cross_track", "same_physical", "none"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-classes", type=int, default=16)
    ap.add_argument("--k-per-class", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lambda-ct", type=float, default=1.0)
    ap.add_argument("--lambda-ce", type=float, default=0.3)
    ap.add_argument("--lambda-temp", type=float, default=0.2)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260814)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--train-classes", default="all",
                    choices=["all", "meta_train", "meta_dev"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    sources = args.source.split(",") if args.source != "mixed" else ["real", "episodic"]
    sampler = PairSampler(sources, args.seed)
    if args.train_classes == "meta_train":
        allowed = class_sets()[0]
        sampler.insts = [x for x in sampler.insts if x["cat"] in allowed]
        sampler.by_cat = defaultdict(list)
        for x in sampler.insts:
            sampler.by_cat[x["cat"]].append(x)
    elif args.train_classes == "meta_dev":
        allowed = class_sets()[1]
        sampler.insts = [x for x in sampler.insts if x["cat"] in allowed]
        sampler.by_cat = defaultdict(list)
        for x in sampler.insts:
            sampler.by_cat[x["cat"]].append(x)
    model = TSR(arch=args.arch).to(args.device)
    cls_head = nn.Linear(model.hidden, 48).to(args.device)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(cls_head.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    all_cats = torch.tensor(sorted(sampler.by_cat), device=args.device)
    cat_index = {int(c): i for i, c in enumerate(sorted(sampler.by_cat))}

    log_path = out / "train.log"
    t0 = __import__("time").time()
    step = 0
    ema_loss = None
    with open(log_path, "w") as logf:
        for step in range(1, args.steps + 1):
            batch, feats, qs, cats, temp_pairs, lens1, lens2 = sampler.sample_batch(
                args.n_classes, args.k_per_class, args.max_len)
            # pad to the maximum prefix length (both l1 and l2 prefixes)
            pad_T = max(max(lens1), max(lens2) if lens2 else 0)
            fpad, qpad, mask = build_padded_batch(
                feats, qs, [min(max(l1, l2), pad_T) for l1, l2 in zip(lens1, lens2)],
                args.device)
            states = model.embed_batch(fpad, qpad, mask)
            s = states[torch.arange(len(batch), device=args.device),
                       torch.tensor(lens1, device=args.device) - 1]
            cats_t = torch.tensor(cats, device=args.device)
            loss = torch.zeros((), device=args.device)
            if args.supervision == "cross_track":
                loss = loss + args.lambda_ct * supcon_loss(s, cats_t, args.temp)
            elif args.supervision == "same_physical":
                if temp_pairs:
                    idx = [p[0] for p in temp_pairs]
                    s1 = s[idx]
                    l2s = torch.tensor([p[2] for p in temp_pairs], device=args.device)
                    s2 = states[torch.tensor(idx, device=args.device), l2s - 1]
                    s_all = torch.cat([s1, s2], dim=0)
                    # pairs (i, i+n) are same instance
                    n = len(s1)
                    sim = s_all @ s_all.t() / args.temp
                    pos_mask = torch.zeros_like(sim, dtype=torch.bool)
                    for i in range(n):
                        pos_mask[i, i + n] = True
                        pos_mask[i + n, i] = True
                    exp = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
                    denom = exp.sum(dim=1, keepdim=True)
                    loss = loss + args.lambda_ct * (
                        -torch.log(exp / (denom + 1e-9) + 1e-12)[pos_mask].mean())
            if args.lambda_temp > 0 and temp_pairs and args.supervision == "cross_track":
                idx = torch.tensor([p[0] for p in temp_pairs], device=args.device)
                l2s = torch.tensor([p[2] for p in temp_pairs], device=args.device)
                s1 = s[idx]
                s2 = states[idx, l2s - 1]
                loss = loss + args.lambda_temp * (1.0 - F.cosine_similarity(s1, s2, dim=-1)).mean()
            if args.lambda_ce > 0:
                logits = cls_head(s)
                targets = torch.tensor([cat_index[c] for c in cats], device=args.device)
                loss = loss + args.lambda_ce * F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ema_loss = float(loss) if ema_loss is None else 0.99 * ema_loss + 0.01 * float(loss)
            if step % 100 == 0:
                line = (f"step {step} loss {float(loss):.4f} ema {ema_loss:.4f} "
                        f"{__import__('time').time()-t0:.0f}s")
                print(line, flush=True)
                logf.write(line + "\n")
                logf.flush()
    torch.save({
        "model": model.state_dict(),
        "cls": cls_head.state_dict(),
        "arch": args.arch,
        "args": vars(args),
    }, out / "checkpoint.pth")
    (out / "train_meta.json").write_text(json.dumps(
        {"steps": args.steps, "loss_ema": ema_loss, "args": vars(args)}))
    print("done", out / "checkpoint.pth")


if __name__ == "__main__":
    main()
