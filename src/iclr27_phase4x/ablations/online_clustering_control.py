"""Reviewer control: online nearest-prototype / DP-means-style clustering.

Known if best known cosine >= tau_k; else existing novel if best novel
cosine >= tau_n; else birth. No noise hypothesis, no posterior.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)
from src.iclr27_phase4x.evaluation.pilot_x3 import load_tsr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau-k", default="0.6,0.7,0.8")
    ap.add_argument("--tau-n", default="0.6,0.7,0.8")
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_dev_categories"]
    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = F.normalize(torch.from_numpy(d["means"]).to(args.device), dim=-1)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    eps = [make_episode(store, pool, cfg, rng) for _ in range(args.n_episodes)]
    rows = {}
    for tk_s in args.tau_k.split(","):
        for tn_s in args.tau_n.split(","):
            tk, tn = float(tk_s), float(tn_s)
            agg = defaultdict(int)
            slots = []
            for ep in eps:
                active_idx = [cat_index[c] for c in ep["pseudo_known"]]
                known = anchors[active_idx]
                novel = torch.zeros(0, 256, device=args.device)
                slot_cat = {}
                stats = defaultdict(int)
                for occ in ep["occurrences"]:
                    z, q = store.tracklet_seq(occ["key"])
                    n = min(len(z), cfg.max_len)
                    state = tsr.init_state(1, args.device)
                    commit = None
                    for t in range(cfg.min_commit_age - 1, n):
                        f = torch.from_numpy(z[t:t + 1]).to(args.device)
                        qt = torch.from_numpy(q[t:t + 1]).to(args.device)
                        s, state = tsr.step(f, qt, state)
                        sn = F.normalize(s, dim=-1)
                        ck = (sn @ known.t())[0]
                        if float(ck.max()) >= tk:
                            commit = ("known", ep["pseudo_known"][int(ck.argmax())], t)
                        elif novel.shape[0] > 0:
                            cn = (sn @ novel.t())[0]
                            if float(cn.max()) >= tn:
                                commit = ("existing", int(cn.argmax()), t)
                            else:
                                commit = ("new", novel.shape[0], t)
                                novel = torch.cat([novel, sn], 0)
                                slot_cat[occ["category"]] = novel.shape[0] - 1
                        else:
                            commit = ("new", novel.shape[0], t)
                            novel = torch.cat([novel, sn], 0)
                            slot_cat[occ["category"]] = novel.shape[0] - 1
                        break
                    cat = occ["category"]
                    if occ["role"] == "known":
                        stats["known_total"] += 1
                        if commit is not None and commit[0] == "known" and commit[1] == cat:
                            stats["known_correct"] += 1
                    elif occ["role"] == "novel":
                        if occ.get("first"):
                            stats["first_total"] += 1
                            stats["first_correct"] += int(commit is not None and commit[0] == "new")
                            stats["absorbed"] += int(commit is not None and commit[0] == "known")
                        else:
                            stats["later_total"] += 1
                            stats["reuse_correct"] += int(
                                commit is not None and commit[0] == "existing"
                                and commit[1] == slot_cat.get(cat, -1))
                            stats["overbirth"] += int(commit is not None and commit[0] == "new")
                    else:
                        stats["fp_born"] += int(commit is not None and commit[0] == "new")
                for kk, v in stats.items():
                    agg[kk] += v
                slots.append(novel.shape[0])
            rows[f"{tk_s}|{tn_s}"] = {
                "known_rate": round(agg["known_correct"] / max(agg["known_total"], 1), 4),
                "first_rate": round(agg["first_correct"] / max(agg["first_total"], 1), 4),
                "reuse_rate": round(agg["reuse_correct"] / max(agg["later_total"], 1), 4),
                "fp_born": agg["fp_born"], "overbirth": agg["overbirth"],
                "absorbed": agg["absorbed"],
                "mean_slots": round(float(np.mean(slots)), 3),
            }
            print(rows[f"{tk_s}|{tn_s}"], flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "control.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
