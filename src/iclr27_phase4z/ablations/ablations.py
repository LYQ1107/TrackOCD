"""Ablations on category-disjoint meta-dev (frozen candidate only):
- temporal-order shuffle (GRU; order must matter for the claimed mechanism)
- prefix-length truncation (evidence accumulation curve)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4z.training.train_routing import (
    GRURouter,
    MLPRouter,
    evaluate_online,
    load_episodes,
    routing_metrics,
)


def load_router(path: Path, device: str):
    ck = torch.load(path / "router.pth", map_location=device)
    mode = ck["mode"]
    if mode == "gru":
        model = GRURouter(283, ck["args"].get("hidden", 96)).to(device)
    else:
        in_dim = {"static": 283, "meanpool": 283, "singleframe": 283,
                  "aggregated": 27 * 4 + 27 + 256 + 256 + 1}[mode]
        model = MLPRouter(in_dim, ck["args"].get("hidden", 128)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mode, ck


def shuffled(seqs, seed):
    rng = random.Random(seed)
    out = []
    for sq in seqs:
        n = sq["len"]
        idx = list(range(n))
        rng.shuffle(idx)
        out.append({
            "ev": sq["ev"][idx], "h": sq["h"][idx],
            "role": sq["role"], "len": n,
        })
    return out


def truncated(seqs, max_age):
    out = []
    for sq in seqs:
        n = min(sq["len"], max_age)
        out.append({
            "ev": sq["ev"][:n], "h": sq["h"][:n],
            "role": sq["role"], "len": n,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True)
    ap.add_argument("--meta-dev-episodes", required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, mode, ck = load_router(ROOT / args.router, args.device)
    seqs = load_episodes(ROOT / args.meta_dev_episodes)
    if ck["args"].get("normalize", 1):
        for sq in seqs:
            sq["ev"] = (sq["ev"] - ck["ev_mean"]) / ck["ev_std"]
            sq["h"] = (sq["h"] - ck["h_mean"]) / ck["h_std"]
    base = routing_metrics(evaluate_online(model, seqs, mode, args.device, args.tau))

    rows = []
    if mode == "gru":
        for seed in (1, 2, 3):
            sh = shuffled(seqs, seed)
            m = routing_metrics(evaluate_online(model, sh, mode, args.device, args.tau))
            m["seed"] = seed
            rows.append(m)

    prefix_rows = []
    for a in (1, 2, 3, 4, 6, 8, 12):
        tr = truncated(seqs, a)
        m = routing_metrics(evaluate_online(model, tr, mode, args.device, args.tau))
        m["max_age"] = a
        prefix_rows.append(m)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.json").write_text(json.dumps({
        "base": base, "shuffled": rows, "prefix_length": prefix_rows,
        "mode": mode, "tau": args.tau,
    }, indent=2))
    print(json.dumps({"base": base, "shuffled": rows, "prefix": prefix_rows},
                     indent=2))


if __name__ == "__main__":
    main()
