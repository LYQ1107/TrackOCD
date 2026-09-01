"""Simple episode-conditioned OOD + nearest-prototype clustering control.

Cold: KNOWN if active-known energy > tau, else NEW (no NO_COMMIT).
Warm: same known rule; NOVEL -> EXISTING if max_novel >= l2_new else NEW.
Model-in-the-loop memory; thresholds swept on TRAIN-only meta-dev.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    load_known_branch,
    load_novel_branch,
    proto_evidence,
)
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_active_universe,
    load_store,
    make_episode,
)


def run(ep, store, cfg, ktsr, kcls, ntsr, l2, protos, cat_index, tau, device):
    active_idx = [cat_index[c] for c in ep["pseudo_known"]]
    memory = NovelMemory(device)
    slot_cat = {}
    cold = defaultdict(int)
    warm = defaultdict(int)
    fp_first = 0
    outcomes = []
    for occ in ep["occurrences"]:
        phase_at_start = memory.size() == 0
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
        commit = None
        for t in range(cfg.min_commit_age - 1, n):
            f = torch.from_numpy(z[t:t + 1]).to(device)
            qt = torch.from_numpy(q[t:t + 1]).to(device)
            rs = float(np.clip(q[t, 0], 0.05, 0.95))
            ev, s_k, s_n, nl, l2_new = ds.step(f, qt, rs, t + 1, memory)
            pe = proto_evidence(s_k, protos, active_idx, tau=0.1)
            if pe[3] > tau:  # active energy high -> KNOWN
                kl = kcls(s_k)[0]
                commit = ("known", known_list[int(kl.argmax())], t)
            elif memory.size() == 0 or nl.shape[1] == 0 or nl.max() < l2_new[0, 0]:
                commit = ("new", memory.size(), t)
                memory.create(s_n, rs, {"cat": occ["category"]})
                slot_cat[occ["category"]] = memory.size() - 1
            else:
                slot = int(nl.argmax())
                commit = ("existing", slot, t)
                memory.update(slot, s_n, rs)
            break
        outcomes.append((occ, commit, phase_at_start))
    # score (no NO_COMMIT in this baseline)
    for occ, commit, phase_cold in outcomes:
        if occ["role"] == "known":
            key = "cold" if phase_cold else "warm"
            (cold if key == "cold" else warm)["known_total"] += 1
            if commit is not None and commit[0] == "known" and commit[1] == occ["category"]:
                (cold if key == "cold" else warm)["known_correct"] += 1
        elif occ["role"] == "novel":
            if occ.get("first"):
                key = "cold" if phase_cold else "warm"
                (cold if key == "cold" else warm)["new_total"] += 1
                if commit is not None and commit[0] == "new":
                    (cold if key == "cold" else warm)["new_correct"] += 1
                elif commit is not None and commit[0] == "known":
                    (cold if key == "cold" else warm)["absorbed"] += 1
            else:
                warm["existing_total"] += 1
                if (commit is not None and commit[0] == "existing"
                        and commit[1] == slot_cat.get(occ["category"], -1)):
                    warm["existing_correct"] += 1
                elif commit is not None and commit[0] == "new":
                    warm["overbirth"] += 1
                elif commit is not None and commit[0] == "known":
                    warm["absorbed"] += 1
                else:
                    warm["wrong_reuse"] += 1
        else:
            if commit is not None and commit[0] == "new":
                if phase_cold:
                    fp_first += 1
                warm["fp_born"] += 1
            elif commit is not None:
                warm["fp_known_commit"] += 1
    return cold, warm, fp_first, memory.size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", default="2.8,3.0,3.2,3.4,3.6,3.8,4.0")
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_dev_categories"]
    protos, cat_index, _ = load_active_universe(args.device)
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    global known_list
    known_list = sorted(known_ids())
    eps = [make_episode(store, pool, cfg, rng) for _ in range(args.n_episodes)]
    rows = {}
    for tau_s in args.thresholds.split(","):
        tau = float(tau_s)
        ac = defaultdict(int); aw = defaultdict(int)
        fpf = 0; slots = []
        for ep in eps:
            c, w, f, k = run(ep, store, cfg, ktsr, kcls, ntsr, l2,
                             protos, cat_index, tau, args.device)
            for kk, v in c.items(): ac[kk] += v
            for kk, v in w.items(): aw[kk] += v
            fpf += f; slots.append(k)
        rows[tau_s] = {
            "cold_known_acc": round(ac["known_correct"] / max(ac["known_total"], 1), 4),
            "cold_new_recall": round(ac["new_correct"] / max(ac["new_total"], 1), 4),
            "cold_absorbed": ac["absorbed"],
            "warm_known_acc": round(aw["known_correct"] / max(aw["known_total"], 1), 4),
            "warm_new_recall": round(aw["new_correct"] / max(aw["new_total"], 1), 4),
            "existing_recall": round(aw["existing_correct"] / max(aw["existing_total"], 1), 4),
            "overbirth": aw["overbirth"],
            "fp_born": aw["fp_born"],
            "fp_born_before_first": fpf,
            "mean_slots": round(float(np.mean(slots)), 3),
        }
        print(tau_s, rows[tau_s], flush=True)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
