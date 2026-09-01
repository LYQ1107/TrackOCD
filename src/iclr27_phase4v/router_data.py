"""Build router training samples from real tracker-induced episodes.

Default: one sample per occurrence per step (matches the first-commit
inference policy used by pilot/dev). With --final-only, keep the old
protocol (one sample per occurrence final step). Memory writes are
teacher-forced from the episode GT (novel first -> create, novel later ->
update), so the router sees a clean causal novel-memory context.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4t.episodes import (
    RealEpisodeConfig,
    RealStreamStore,
    make_real_episode,
)
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    load_known_branch,
    load_novel_branch,
)


def load_store():
    rows = list(csv.DictReader(open(ROOT / "outputs/iclr27_phase4t/train_stream/proposals.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"]); r["frame_id"] = int(r["frame_id"])
        r["track_id"] = int(r["track_id"]); r["score"] = float(r["score"])
        r["q_phys"] = json.loads(r["q_phys"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["gt_role"] = r["gt_role"]; r["gt_category_id"] = int(r["gt_category_id"])
        r["gt_iou"] = float(r["gt_iou"]); r["gt_track_id"] = int(r["gt_track_id"])
        r["prior_hits"] = int(r["prior_hits"]); r["age"] = int(r["age"])
        r["gap"] = int(r["gap"]); r["run_score_mean"] = float(r["run_score_mean"])
    feats = np.load(ROOT / "outputs/iclr27_phase4t/train_stream/feats.npz")["feats"]
    return RealStreamStore(rows, feats)


def build_samples(n_episodes: int, seed: int, device: str,
                  max_len: int = 8, store=None, all_steps: bool = False):
    store = store or load_store()
    ktsr, kcls = load_known_branch(device)
    ntsr, l2 = load_novel_branch(device)
    cfg = RealEpisodeConfig()
    rng = random.Random(seed)
    X, y, roles, cats, ep_idx = [], [], [], [], []
    known_list = sorted(known_ids())
    known_index = {c: i for i, c in enumerate(known_list)}
    for e in range(n_episodes):
        ep = make_real_episode(store, cfg, rng)
        ep_known_idx = [known_index[c] for c in ep["pseudo_known"]]
        memory = NovelMemory(device)
        slot_cat = {}
        for occ in ep["occurrences"]:
            z, q = store.tracklet_seq(occ["key"])
            n = min(len(z), max_len)
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
            created = False
            for t in range(n):
                f = torch.from_numpy(z[t:t + 1]).to(device)
                qt = torch.from_numpy(q[t:t + 1]).to(device)
                r_scalar = float(np.clip(q[t, 0], 0.05, 0.95))
                ev, s_k, s_n, nl, l2_new = ds.step(
                    f, qt, r_scalar, t + 1, memory, known_idx=ep_known_idx)
                if all_steps or t == n - 1:
                    label = 1 if occ["role"] == "known" else (
                        0 if occ["role"] == "novel" else 2)
                    X.append(ev)
                    y.append(label)
                    roles.append(occ["role"])
                    cats.append(occ["category"])
                    ep_idx.append(e)
                # teacher-forced memory writes from commit age
                if occ["role"] == "novel" and t >= cfg.min_commit_age - 1:
                    c = occ["category"]
                    if occ.get("first") and not created:
                        memory.create(s_n, float(q[t, 0]), {"cat": c})
                        slot_cat[c] = memory.size() - 1
                        created = True
                    elif c in slot_cat:
                        memory.update(slot_cat[c], s_n, float(q[t, 0]))
        if (e + 1) % 50 == 0:
            print(f"episode {e+1}/{n_episodes}", flush=True)
    return (np.stack(X).astype(np.float32),
            np.asarray(y, dtype=np.int64),
            roles, cats, ep_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--all-steps", action="store_true",
                    help="sample every step (matches first-commit inference)")
    args = ap.parse_args()
    X, y, roles, cats, ep_idx = build_samples(
        args.n_episodes, args.seed, args.device, all_steps=args.all_steps)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", X=X, y=y, ep_idx=np.asarray(ep_idx))
    (out / "meta.json").write_text(json.dumps({
        "n": len(y), "n_episodes": args.n_episodes, "seed": args.seed,
        "all_steps": bool(args.all_steps),
        "dim": int(X.shape[1]),
        "label_counts": {str(k): int(v) for k, v in
                         zip(*np.unique(y, return_counts=True))},
    }, indent=2))
    print("saved", out / "samples.npz", X.shape)


if __name__ == "__main__":
    main()
