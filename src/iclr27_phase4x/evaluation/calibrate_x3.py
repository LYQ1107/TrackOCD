"""Calibrate X3 hyperparameters on TRAIN-only meta-dev episodes."""
from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict

import numpy as np
import torch

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr, run_episode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="outputs/iclr27_phase4x/simple_mixture/calibration.json")
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_dev_categories"]
    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    eps = [make_episode(store, pool, cfg, rng) for _ in range(args.n_episodes)]

    grid = list(itertools.product(
        [2.0, 4.0, 8.0, 16.0],       # kappa
        [-1.5, -0.5],                # log_prior_new
        [2.0, 4.0],                  # noise_alpha
        [1.5, 2.0],                  # margin_ratio
    ))
    rows = []
    for kappa, lpn, na, mr in grid:
        hp = {"kappa": kappa, "log_prior_new": lpn, "log_prior_noise": -3.0,
              "noise_alpha": na, "commit_threshold": 0.5, "margin_ratio": mr,
              "min_age": 2}
        agg = defaultdict(int)
        slots = []
        for ep in eps:
            st, k = run_episode(ep, store, cfg, tsr, anchors, cat_index, hp,
                                args.device)
            for kk, v in st.items():
                agg[kk] += v
            slots.append(k)
        known_r = agg["known_correct"] / max(agg["known_total"], 1)
        first_r = agg["first_correct"] / max(agg["first_total"], 1)
        reuse_r = agg["reuse_correct"] / max(agg["later_total"], 1)
        fp_r = agg["fp_born"] / max(agg["fp_born"] + agg["fp_no_write"] +
                                    agg["fp_unresolved"] + agg["fp_other_commit"], 1)
        obj = (first_r + reuse_r) / 2
        rows.append({"hparams": hp, "known_rate": round(known_r, 4),
                     "first_rate": round(first_r, 4),
                     "reuse_rate": round(reuse_r, 4),
                     "fp_born": agg["fp_born"], "fp_rate": round(fp_r, 4),
                     "overbirth": agg["overbirth"],
                     "absorbed": agg["absorbed"],
                     "mean_slots": round(float(np.mean(slots)), 3),
                     "objective": round(obj, 4)})
        print(rows[-1], flush=True)
    valid = [r for r in rows if r["fp_born"] <= 60 and r["known_rate"] >= 0.35]
    best = max(valid, key=lambda r: r["objective"]) if valid else max(rows, key=lambda r: r["objective"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "best": best}, indent=2))
    print("BEST", json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
