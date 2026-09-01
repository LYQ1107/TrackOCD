"""Model-in-the-loop episodic pilot for Phase 4W.

State machine: K==0 -> ColdStartHead (KNOWN/NEW/NO_COMMIT); K>0 ->
WarmMemoryHead (KNOWN/NEW/EXISTING/NO_COMMIT). Memory evolves from the
model's own prior actions (no teacher forcing in evaluation).
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
from src.iclr27_phase4w.cold_start.train import ColdStartHead, WarmMemoryHead
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_active_universe,
    load_store,
    make_episode,
)


def load_heads(device):
    c = torch.load(ROOT / "outputs/iclr27_phase4w/cold_start/head_cold_v3/head.pth",
                   map_location=device)
    w = torch.load(ROOT / "outputs/iclr27_phase4w/warm_memory/head_warm_v3/head.pth",
                   map_location=device)
    cold = ColdStartHead(c["dim"]).to(device)
    cold.load_state_dict(c["model"]); cold.eval()
    warm = WarmMemoryHead(w["dim"]).to(device)
    warm.load_state_dict(w["model"]); warm.eval()
    return cold, warm


def run_episode(ep, store, cfg, ktsr, kcls, ntsr, l2, protos, cat_index,
                proj_t, cold, warm, device, known_list):
    active_idx = [cat_index[c] for c in ep["pseudo_known"]]
    memory = NovelMemory(device)
    slot_cat = {}
    cold_stats = defaultdict(int)
    warm_stats = defaultdict(int)
    fp_before_first = 0
    outcomes = []
    for occ in ep["occurrences"]:
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
        commit = None
        for t in range(n):
            f = torch.from_numpy(z[t:t + 1]).to(device)
            qt = torch.from_numpy(q[t:t + 1]).to(device)
            rs = float(np.clip(q[t, 0], 0.05, 0.95))
            ev, s_k, s_n, nl, l2_new = ds.step(f, qt, rs, t + 1, memory)
            pe = proto_evidence(s_k, protos, active_idx, tau=0.1)
            skp = (torch.nn.functional.normalize(s_k, dim=-1) @ proj_t)[0]
            skp = skp.detach().cpu().numpy()
            qv = q[t].astype(np.float32)
            if memory.size() == 0:
                x = np.concatenate([pe, skp, qv]).astype(np.float32)
                logits = cold(torch.from_numpy(x).unsqueeze(0).to(device))[0]
                a = int(logits.argmax())
                if a == 0:  # KNOWN
                    kl = kcls(s_k)[0]
                    commit = ("known", known_list[int(kl.argmax())], t)
                    break
                elif a == 1:  # NEW
                    commit = ("new", memory.size(), t)
                    memory.create(s_n, rs, {"cat": occ["category"]})
                    slot_cat[occ["category"]] = memory.size() - 1
                    break
                else:  # NO_COMMIT
                    continue
            else:
                mem_ev = np.concatenate([ev[8:12], qv]).astype(np.float32)
                x = np.concatenate([pe, skp, mem_ev]).astype(np.float32)
                logits = warm(torch.from_numpy(x).unsqueeze(0).to(device))[0]
                a = int(logits.argmax())
                if a == 0:  # KNOWN
                    kl = kcls(s_k)[0]
                    commit = ("known", known_list[int(kl.argmax())], t)
                    break
                elif a == 1:  # NEW
                    commit = ("new", memory.size(), t)
                    memory.create(s_n, rs, {"cat": occ["category"]})
                    slot_cat[occ["category"]] = memory.size() - 1
                    break
                elif a == 2:  # EXISTING
                    if nl.shape[1] >= 1:
                        slot = int(nl.argmax())
                        commit = ("existing", slot, t)
                        memory.update(slot, s_n, rs)
                    else:
                        commit = ("new", memory.size(), t)
                        memory.create(s_n, rs, {"cat": occ["category"]})
                        slot_cat[occ["category"]] = memory.size() - 1
                    break
                else:  # NO_COMMIT
                    continue
        outcomes.append((occ, commit))

    replay = NovelMemory(device)
    for occ, commit in outcomes:
        phase_cold = replay.size() == 0
        cat = occ["category"]
        if occ["role"] == "known":
            if phase_cold:
                cold_stats["known_total"] += 1
                if commit is not None and commit[0] == "known" and commit[1] == cat:
                    cold_stats["known_correct"] += 1
                elif commit is not None and commit[0] == "new":
                    cold_stats["false_new_known"] += 1
            else:
                warm_stats["known_total"] += 1
                if commit is not None and commit[0] == "known" and commit[1] == cat:
                    warm_stats["known_correct"] += 1
        elif occ["role"] == "novel":
            if occ.get("first"):
                if phase_cold:
                    cold_stats["new_total"] += 1
                    if commit is not None and commit[0] == "new":
                        cold_stats["new_correct"] += 1
                    elif commit is not None and commit[0] == "known":
                        cold_stats["absorbed"] += 1
                else:
                    warm_stats["birth_total"] += 1
                    if commit is not None and commit[0] == "new":
                        warm_stats["birth_correct"] += 1
                    elif commit is not None and commit[0] == "known":
                        warm_stats["absorbed"] += 1
            else:
                if phase_cold:
                    cold_stats["later_before_memory"] += 1
                else:
                    warm_stats["existing_total"] += 1
                    if (commit is not None and commit[0] == "existing"
                            and commit[1] == slot_cat.get(cat, -1)):
                        warm_stats["existing_correct"] += 1
                    elif commit is not None and commit[0] == "new":
                        warm_stats["overbirth"] += 1
                    elif commit is not None and commit[0] == "known":
                        warm_stats["absorbed"] += 1
                    else:
                        warm_stats["wrong_reuse"] += 1
        else:  # fp
            if commit is None:
                (cold_stats if phase_cold else warm_stats)["fp_no_commit"] += 1
            elif commit[0] == "new":
                if phase_cold:
                    fp_before_first += 1
                (cold_stats if phase_cold else warm_stats)["fp_born"] += 1
            elif commit[0] == "known":
                (cold_stats if phase_cold else warm_stats)["fp_known_commit"] += 1
            else:
                (cold_stats if phase_cold else warm_stats)["fp_existing_commit"] += 1
        # replay memory writes (mirror of model actions)
        if commit is not None and commit[0] == "new":
            replay.create(torch.zeros(1, 256, device=device), 0.5, {"cat": cat})
        elif commit is not None and commit[0] == "existing":
            replay.update(commit[1], torch.zeros(1, 256, device=device), 0.5)
    return cold_stats, warm_stats, fp_before_first, memory.size()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "metadev"], required=True)
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    store = load_store()
    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    protos, cat_index, proj_t = load_active_universe(args.device)
    ktsr, kcls = load_known_branch(args.device)
    ntsr, l2 = load_novel_branch(args.device)
    cold, warm = load_heads(args.device)
    cfg = WEpisodeConfig()
    rng = random.Random(args.seed)
    known_list = sorted(known_ids())
    agg_c = defaultdict(int)
    agg_w = defaultdict(int)
    fp_first = 0
    slots = []
    for e in range(args.n_episodes):
        ep = make_episode(store, pool, cfg, rng)
        cs, ws, fbf, k = run_episode(ep, store, cfg, ktsr, kcls, ntsr,
                                     l2, protos, cat_index, proj_t,
                                     cold, warm, args.device, known_list)
        for kk, v in cs.items():
            agg_c[kk] += v
        for kk, v in ws.items():
            agg_w[kk] += v
        fp_first += fbf
        slots.append(k)

    def rates(stats):
        out = dict(stats)
        for num, den in [("known_correct", "known_total"),
                         ("new_correct", "new_total"),
                         ("existing_correct", "existing_total"),
                         ("birth_correct", "birth_total")]:
            if den in stats and stats[den] > 0:
                out[num + "_rate"] = round(stats.get(num, 0) / stats[den], 4)
        return out

    report = {
        "split": args.split,
        "n_episodes": args.n_episodes,
        "cold": rates(dict(agg_c)),
        "warm": rates(dict(agg_w)),
        "fp_born_before_first_novel": fp_first,
        "mean_slots": round(float(np.mean(slots)), 3),
    }
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
