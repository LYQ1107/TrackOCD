"""ADSSI meta-dev/train episodic pilot (model-in-the-loop)."""
from __future__ import annotations

import argparse
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
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr
from src.iclr27_phase4y.evaluation.rollout import run_episode_eval
from src.iclr27_phase4y.model import ADSSI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "metadev"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit-threshold", type=float, default=0.5)
    ap.add_argument("--margin-ratio", type=float, default=1.5)
    ap.add_argument("--min-age", type=int, default=2)
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    ck = torch.load(ROOT / args.checkpoint, map_location=args.device)
    model = ADSSI(in_dim=256, d=128).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    agg = defaultdict(int)
    slots = []
    with torch.no_grad():
        for e in range(args.n_episodes):
            ep = make_episode(store, pool, cfg, rng)
            st, k = run_episode_eval(
                model, ep, store, cfg, tsr, anchors, cat_index, args.device,
                args.commit_threshold, args.margin_ratio, args.min_age)
            for kk, v in st.items():
                agg[kk] += v
            slots.append(k)
    report = dict(agg)
    for num, den in [("known_correct", "known_total"),
                     ("first_correct", "first_total"),
                     ("reuse_correct", "later_total")]:
        if den in agg and agg[den]:
            report[num + "_rate"] = round(agg[num] / agg[den], 4)
    report["mean_slots"] = round(float(np.mean(slots)), 3)
    report["checkpoint"] = args.checkpoint
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
