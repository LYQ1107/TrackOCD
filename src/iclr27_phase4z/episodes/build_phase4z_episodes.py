"""Genuine-OOV routing episodes for Phase 4Z (frozen O1c evidence path).

Each episode has an active supported-known subset (pseudo-known) and
pseudo-novel categories excluded from that active subset (Phase4W protocol).
Tracklets come from the real Q1 TRAIN tracker-induced stream. For every
causal prefix step the frozen O1c model (d2_joint_v2 TSR + T3/D2 heads)
produces:
  - active-subset known statistics (top1_p/margin/entropy/energy)
  - full-48 known statistics (aggregate only; no per-class leak)
  - frozen level-1 router probabilities (computed on the active subset)
  - physical evidence q, r, normalized age
  - TSR state h_t

No benchmark novel GT, no dev GT, no full-48 per-class logits for pseudo-novel
tracklets (aggregate statistics only).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4u.downstream.model import (
    HierarchicalTSRCore,
    build_tsr_known_protos,
)
from src.iclr27_phase4u.trajectory.model import TSR
from src.iclr27_phase4z.evidence.step_evidence import step_evidence
from src.iclr27_phase4w.episodes.build_episodes import (
    WEpisodeConfig,
    load_store,
    make_episode,
)


def build_episodes(store, pool, n_episodes, seed, device, model, known_list,
                   full_idx, max_len, fp_per_episode=4, known_set_sizes=None,
                   verbose=True):
    cfg = WEpisodeConfig(fp_per_episode=fp_per_episode, max_len=max_len)
    rng = random.Random(seed)
    sizes = known_set_sizes or [cfg.num_pseudo_known]
    ev_list, h_list, y_list = [], [], []
    seq_start = [0]
    seq_role, seq_ep, seq_active_n = [], [], []
    role_counts = Counter()
    with torch.no_grad():
        for e in range(n_episodes):
            nk = sizes[rng.randrange(len(sizes))]
            ep = make_episode(store, pool, cfg, rng, num_pseudo_known=nk)
            active_idx = [known_list.index(c) for c in ep["pseudo_known"]]
            for occ in ep["occurrences"]:
                z, q = store.tracklet_seq(occ["key"])
                n = min(len(z), max_len)
                if n < 1:
                    continue
                model.begin_occurrence(
                    torch.from_numpy(z[:n]).to(device),
                    torch.from_numpy(q[:n]).to(device))
                if occ["role"] == "known":
                    role = 0
                elif occ["role"] == "novel":
                    role = 1
                else:
                    role = 2
                for t in range(n):
                    zt = model.encode(torch.from_numpy(z[t:t + 1]).to(device))
                    qt = torch.from_numpy(q[t:t + 1]).to(device)
                    r_scalar = float(np.clip(q[t, 0], 0.05, 0.95))
                    age = torch.tensor([[float(t + 1)]], device=device)
                    ev, l1p, kl_f = step_evidence(
                        model, zt, qt, age, q[t], r_scalar, active_idx, full_idx)
                    # Drop the frozen Stage C router probabilities (18:21):
                    # including them lets the new router copy the known-biased
                    # frozen boundary (empirically RR ~ 0.048 at dev).
                    ev = np.concatenate([ev[:18], ev[21:]]).astype(np.float32)
                    ev_list.append(ev)
                    h_list.append(zt[0].cpu().numpy().astype(np.float32))
                    y_list.append(role)
                seq_start.append(seq_start[-1] + n)
                seq_role.append(role)
                seq_ep.append(e)
                seq_active_n.append(len(active_idx))
                role_counts[role] += 1
            if verbose and (e + 1) % 100 == 0:
                print(f"episode {e + 1}/{n_episodes}", flush=True)
    return {
        "ev": np.stack(ev_list).astype(np.float32),
        "h": np.stack(h_list).astype(np.float32),
        "y_role": np.asarray(y_list, dtype=np.int8),
        "seq_start": np.asarray(seq_start, dtype=np.int32),
        "seq_role": np.asarray(seq_role, dtype=np.int8),
        "seq_episode": np.asarray(seq_ep, dtype=np.int32),
        "seq_active_n": np.asarray(seq_active_n, dtype=np.int32),
    }, dict(role_counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "metadev"], required=True)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-len", type=int, default=12)
    ap.add_argument("--known-set-sizes", default=None)
    ap.add_argument("--fp-per-episode", type=int, default=4)
    args = ap.parse_args()

    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    sizes = ([int(x) for x in args.known_set_sizes.split(",")]
             if args.known_set_sizes else None)
    known_list = sorted(known_ids())
    full_idx = list(range(len(known_list)))

    rep = TSR(arch="gru").to(args.device)
    ck = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                    map_location=args.device)
    tsr_sd = {k[len("rep."):]: v for k, v in ck["model"].items() if k.startswith("rep.")}
    rep.load_state_dict(tsr_sd)
    rep.eval()
    protos = build_tsr_known_protos(rep, args.device)
    model = HierarchicalTSRCore(rep, protos, use_defer=False,
                                use_qphys=True).to(args.device)
    model.load_t3_init(str(ROOT / "outputs/iclr27_phase4t/t3/checkpoint.pth"),
                       args.device)
    ck2 = torch.load(ROOT / "outputs/iclr27_phase4u/downstream/d2_joint_v2/checkpoint.pth",
                     map_location=args.device)
    ck2["model"].pop("known_raw", None)
    model.load_state_dict(ck2["model"], strict=False)
    model.eval()

    store = load_store()
    data, role_counts = build_episodes(
        store, pool, args.n_episodes, args.seed, args.device, model, known_list,
        full_idx, args.max_len, args.fp_per_episode, sizes)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "episodes.npz", **data)
    seq_lens = np.diff(data["seq_start"])
    (out / "meta.json").write_text(json.dumps({
        "split": args.split, "pool": pool, "n_episodes": args.n_episodes,
        "seed": args.seed, "max_len": args.max_len,
        "known_set_sizes": sizes or [4],
        "n_seqs": int(len(data["seq_role"])),
        "n_steps": int(len(data["y_role"])),
        "seq_len_mean": float(seq_lens.mean()),
        "seq_len_max": int(seq_lens.max()),
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "ev_dim": int(data["ev"].shape[1]),
        "h_dim": int(data["h"].shape[1]),
        "feature_names": [
            "active_sim_top1", "active_sim_margin", "active_sim_entropy",
            "active_sim_energy",
            "full_sim_top1", "full_sim_margin", "full_sim_entropy",
            "full_sim_energy",
            "active_top1_p", "active_margin", "active_entropy", "active_energy",
            "active_max_kl",
            "full_top1_p", "full_margin", "full_entropy", "full_energy",
            "full_max_kl",
            "active_residual",
            "q0", "q1", "q2", "q3", "q4", "q5", "r", "age_norm",
        ],
    }, indent=2))
    print(json.dumps({
        "split": args.split, "n_seqs": int(len(data["seq_role"])),
        "n_steps": int(len(data["y_role"])),
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "mean_len": float(seq_lens.mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
