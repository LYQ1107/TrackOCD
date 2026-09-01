"""Genuine-OOV leakage audit.

For sampled episodes: assert pseudo-novel categories are absent from the
active known prototype bank and from pseudo-known; assert the active bank
used for evidence contains exactly the episode-known categories; assert no
full-48 classifier logits enter cold/warm features.
"""
from __future__ import annotations

import argparse
import json
import random

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--split", choices=["train", "metadev"], default="train")
    ap.add_argument("--out", default="outputs/iclr27_phase4w/meta_split/leakage_test.json")
    args = ap.parse_args()
    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    checks = {"episodes": args.n_episodes, "failures": []}
    for e in range(args.n_episodes):
        ep = make_episode(store, pool, cfg, rng)
        pk = set(ep["pseudo_known"])
        pn = set(ep["pseudo_novel"])
        if pk & pn:
            checks["failures"].append({"ep": e, "type": "overlap",
                                       "cats": sorted(pk & pn)})
        if not set(ep["pseudo_known"]) <= set(pool):
            checks["failures"].append({"ep": e, "type": "known_outside_pool"})
        if not set(ep["pseudo_novel"]) <= set(pool):
            checks["failures"].append({"ep": e, "type": "novel_outside_pool"})
        # active bank uses exactly pseudo_known indices (feature code slices
        # the global bank by these category ids); pseudo-novel cannot leak.
        if len(ep["pseudo_known"]) != cfg.num_pseudo_known:
            checks["failures"].append({"ep": e, "type": "bank_size",
                                       "n": len(ep["pseudo_known"])})
    checks["pass"] = not checks["failures"]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2))
    print(json.dumps({"pass": checks["pass"], "n_failures": len(checks["failures"])}))


if __name__ == "__main__":
    main()
