"""X3 model-in-the-loop episodic pilot (vMF components + sequential posterior)."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)
from src.iclr27_phase4x.components.vmf_memory import (
    CompatSemanticMemory,
    VMFSemanticMemory,
)
from src.iclr27_phase4x.likelihood.train_compatibility import CompatibilityNet


def load_tsr(device):
    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=device)
    sd = {k[len("rep."):]: v for k, v in ck["model"].items() if k.startswith("rep.")}
    tsr = TSR(arch="gru").to(device)
    tsr.load_state_dict(sd)
    tsr.eval()
    return tsr


def run_episode(ep, store, cfg, tsr, anchors, cat_index, hparams, device,
                compat=None):
    active_idx = [cat_index[c] for c in ep["pseudo_known"]]
    if compat is None:
        mem = VMFSemanticMemory(anchors, kappa=hparams["kappa"],
                                log_prior_new=hparams["log_prior_new"],
                                log_prior_noise=hparams["log_prior_noise"],
                                noise_alpha=hparams["noise_alpha"],
                                device=device)
    else:
        mem = CompatSemanticMemory(anchors, compat,
                                   log_prior_new=hparams["log_prior_new"],
                                   log_prior_noise=hparams["log_prior_noise"],
                                   noise_alpha=hparams["noise_alpha"],
                                   device=device)
    stats = defaultdict(int)
    slot_cat = {}
    outcomes = []
    for occ in ep["occurrences"]:
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        state = tsr.init_state(1, device)
        commit = None
        for t in range(n):
            f = torch.from_numpy(z[t:t + 1]).to(device)
            qt = torch.from_numpy(q[t:t + 1]).to(device)
            rs = float(np.clip(q[t, 0], 0.05, 0.95))
            s, state = tsr.step(f, qt, state)
            if t < hparams["min_age"]:
                continue
            post, _, info = mem.posterior(s, float(q[t, 0]), active_idx)
            p = post[0]
            top2 = torch.topk(p, k=2).values
            if float(p.max()) < hparams["commit_threshold"]:
                continue
            if float(top2[0] / max(top2[1], 1e-9)) < hparams["margin_ratio"]:
                continue
            a = int(p.argmax())
            if a < len(active_idx):
                commit = ("known", ep["pseudo_known"][a], t)
            elif a < len(active_idx) + mem.size():
                k = a - len(active_idx)
                commit = ("existing", k, t)
                mem.update(k, s, rs)
            elif a == len(active_idx) + mem.size():
                k = mem.create(s, rs)
                slot_cat[occ["category"]] = k
                commit = ("new", k, t)
            else:
                commit = ("noise", None, t)
            break
        outcomes.append((occ, commit))
    for occ, commit in outcomes:
        cat = occ["category"]
        if occ["role"] == "known":
            stats["known_total"] += 1
            if commit is not None and commit[0] == "known" and commit[1] == cat:
                stats["known_correct"] += 1
            elif commit is not None and commit[0] == "new":
                stats["known_to_new"] += 1
            elif commit is not None and commit[0] == "existing":
                stats["known_to_existing"] += 1
            elif commit is not None and commit[0] == "noise":
                stats["known_to_noise"] += 1
            else:
                stats["known_unresolved"] += 1
        elif occ["role"] == "novel":
            if occ.get("first"):
                stats["first_total"] += 1
                if commit is not None and commit[0] == "new":
                    stats["first_correct"] += 1
                elif commit is not None and commit[0] == "known":
                    stats["absorbed"] += 1
                elif commit is not None and commit[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif commit is not None and commit[0] == "noise":
                    stats["first_to_noise"] += 1
                else:
                    stats["first_unresolved"] += 1
            else:
                stats["later_total"] += 1
                if (commit is not None and commit[0] == "existing"
                        and commit[1] == slot_cat.get(cat, -1)):
                    stats["reuse_correct"] += 1
                elif commit is not None and commit[0] == "new":
                    stats["overbirth"] += 1
                elif commit is not None and commit[0] == "known":
                    stats["absorbed"] += 1
                elif commit is not None and commit[0] == "existing":
                    stats["wrong_reuse"] += 1
                elif commit is not None and commit[0] == "noise":
                    stats["later_to_noise"] += 1
                else:
                    stats["later_unresolved"] += 1
        else:
            if commit is not None and commit[0] == "new":
                stats["fp_born"] += 1
            elif commit is not None and commit[0] == "noise":
                stats["fp_no_write"] += 1
            elif commit is not None:
                stats["fp_other_commit"] += 1
            else:
                stats["fp_unresolved"] += 1
    return stats, mem.size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "metadev"], required=True)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kappa", type=float, default=32.0)
    ap.add_argument("--log-prior-new", type=float, default=-5.0)
    ap.add_argument("--log-prior-noise", type=float, default=-3.0)
    ap.add_argument("--noise-alpha", type=float, default=2.0)
    ap.add_argument("--commit-threshold", type=float, default=0.5)
    ap.add_argument("--margin-ratio", type=float, default=2.0)
    ap.add_argument("--min-age", type=int, default=2)
    ap.add_argument("--compat", default=None,
                    help="path to learned compatibility checkpoint (X4)")
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    tsr = load_tsr(args.device)
    d = np.load(ROOT / "outputs/iclr27_phase4x/simple_mixture/known_anchors.npz")
    anchors = torch.from_numpy(d["means"]).to(args.device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    hparams = {k: getattr(args, k) for k in
               ("kappa", "log_prior_new", "log_prior_noise", "noise_alpha",
                "commit_threshold", "margin_ratio", "min_age")}
    compat = None
    if args.compat:
        ck = torch.load(ROOT / args.compat, map_location=args.device)
        compat = CompatibilityNet().to(args.device)
        compat.load_state_dict(ck["model"])
        compat.eval()
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    agg = defaultdict(int)
    slots = []
    for e in range(args.n_episodes):
        ep = make_episode(store, pool, cfg, rng)
        st, k = run_episode(ep, store, cfg, tsr, anchors, cat_index, hparams,
                            args.device, compat=compat)
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
    report["hparams"] = hparams
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
