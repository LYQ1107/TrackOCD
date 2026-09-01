"""Episode builder + cold/warm sample generator for Phase 4W.

Genuine OOV: episode-known categories are the ONLY categories in the
active prototype bank for that episode; pseudo-novel categories are
completely absent (asserted). Category pools are disjoint between
meta-train and meta-dev (TRAIN-only 48 categories).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.model import NovelMemory
from src.iclr27_phase4s.protocol import known_ids
from src.iclr27_phase4t.episodes import RealEpisodeConfig, RealStreamStore
from src.iclr27_phase4t.stream_data import build_tracklets
from src.iclr27_phase4u.data import ROOT
from src.iclr27_phase4v.evidence import (
    DualSpaceStep,
    known_evidence,
    load_known_branch,
    load_novel_branch,
    proto_evidence,
)


@dataclass
class WEpisodeConfig:
    num_pseudo_known: int = 4
    num_pseudo_novel: int = 3
    tracklets_per_novel: int = 3
    tracklets_per_known: int = 2
    fp_per_episode: int = 4
    min_commit_age: int = 2
    max_len: int = 8
    seed: int = 20260815


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
    store = RealStreamStore(rows, feats)
    return store


def load_active_universe(device):
    d = np.load(ROOT / "outputs/iclr27_phase4w/active_universe/active_protos.npz")
    protos = torch.from_numpy(d["protos"]).to(device)
    cat_ids = d["cat_ids"].tolist()
    cat_index = {c: i for i, c in enumerate(cat_ids)}
    proj = np.load(ROOT / "outputs/iclr27_phase4w/active_universe/sk_proj.npz")["proj"]
    proj_t = torch.from_numpy(proj).to(device)
    return protos, cat_index, proj_t


def make_episode(store, pool: list[int], cfg: WEpisodeConfig,
                 rng: random.Random,
                 num_pseudo_known: int | None = None) -> dict:
    num_pseudo_known = num_pseudo_known or cfg.num_pseudo_known
    cats = [c for c in pool if len(store.by_cat.get(c, [])) >= 2]
    if len(cats) < num_pseudo_known + cfg.num_pseudo_novel:
        raise ValueError(f"pool too small: {len(cats)}")
    rng.shuffle(cats)
    pseudo_novel = cats[: cfg.num_pseudo_novel]
    pseudo_known = cats[cfg.num_pseudo_novel: cfg.num_pseudo_novel + num_pseudo_known]
    assert not (set(pseudo_known) & set(pseudo_novel))

    occurrences = []
    for c in pseudo_novel:
        keys = store.by_cat[c][:]
        rng.shuffle(keys)
        picked = []
        vids = {}
        for k in keys:
            vids.setdefault(k[0], []).append(k)
        for vk in vids.values():
            picked.append(vk[0])
            if len(picked) >= cfg.tracklets_per_novel:
                break
        if len(picked) < cfg.tracklets_per_novel:
            picked.extend(keys[: cfg.tracklets_per_novel - len(picked)])
        for k in picked[: cfg.tracklets_per_novel]:
            occurrences.append({"key": k, "role": "novel", "category": c})
    for c in pseudo_known:
        keys = store.by_cat[c][:]
        rng.shuffle(keys)
        for k in keys[: cfg.tracklets_per_known]:
            occurrences.append({"key": k, "role": "known", "category": c})
    fp_pool = [k for k in store.fp_tracklets if store.tracklets[k]["length"] >= 2]
    rng.shuffle(fp_pool)
    for k in fp_pool[: cfg.fp_per_episode]:
        occurrences.append({"key": k, "role": "fp", "category": -1})
    rng.shuffle(occurrences)
    seen = set()
    for occ in occurrences:
        if occ["role"] == "novel":
            occ["first"] = occ["category"] not in seen
            seen.add(occ["category"])
    return {"pseudo_known": pseudo_known, "pseudo_novel": pseudo_novel,
            "occurrences": occurrences}


def build_samples(n_episodes: int, seed: int, pool: list[int], device: str,
                  cfg: WEpisodeConfig | None = None,
                  known_set_sizes: list[int] | None = None,
                  evidence_mode: str = "active"):
    cfg = cfg or WEpisodeConfig()
    store = load_store()
    protos, cat_index, proj_t = load_active_universe(device)
    ktsr, kcls = load_known_branch(device)
    ntsr, l2 = load_novel_branch(device)
    rng = random.Random(seed)
    sizes = known_set_sizes or [cfg.num_pseudo_known]
    cold_X, cold_y, warm_X, warm_y = [], [], [], []
    roles = []
    for e in range(n_episodes):
        nk = sizes[rng.randrange(len(sizes))]
        ep = make_episode(store, pool, cfg, rng, num_pseudo_known=nk)
        active_idx = [cat_index[c] for c in ep["pseudo_known"]]
        memory = NovelMemory(device)
        slot_cat = {}
        for occ in ep["occurrences"]:
            z, q = store.tracklet_seq(occ["key"])
            n = min(len(z), cfg.max_len)
            ds = DualSpaceStep(ktsr, kcls, ntsr, l2, device)
            for t in range(n):
                f = torch.from_numpy(z[t:t + 1]).to(device)
                qt = torch.from_numpy(q[t:t + 1]).to(device)
                rs = float(np.clip(q[t, 0], 0.05, 0.95))
                ev, s_k, s_n, nl, l2_new = ds.step(f, qt, rs, t + 1, memory)
                if evidence_mode == "active":
                    pe = proto_evidence(s_k, protos, active_idx, tau=0.1)
                else:
                    pe = known_evidence(kcls(s_k))
                skp = (torch.nn.functional.normalize(s_k, dim=-1) @ proj_t)[0]
                skp = skp.detach().cpu().numpy()
                qv = q[t].astype(np.float32)
                if memory.size() == 0:
                    if t < cfg.min_commit_age - 1:
                        label = 2  # NO_COMMIT until the decision age
                    else:
                        label = 0 if occ["role"] == "known" else (
                            1 if occ["role"] == "novel" else 2)
                    cold_X.append(np.concatenate([pe, skp, qv]))
                    cold_y.append(label)
                else:
                    if t < cfg.min_commit_age - 1:
                        label = 3  # NO_COMMIT
                    elif occ["role"] == "known":
                        label = 0
                    elif occ["role"] == "novel":
                        label = 1 if occ.get("first") else 2
                    else:
                        label = 3
                    mem_ev = np.concatenate([ev[8:12], qv])
                    warm_X.append(np.concatenate([pe, skp, mem_ev]))
                    warm_y.append(label)
                # teacher-forced memory writes
                if occ["role"] == "novel" and t >= cfg.min_commit_age - 1:
                    c = occ["category"]
                    if occ.get("first"):
                        memory.create(s_n, rs, {"cat": c})
                        slot_cat[c] = memory.size() - 1
                    elif c in slot_cat:
                        memory.update(slot_cat[c], s_n, rs)
        roles.append(occ["role"])
        if (e + 1) % 100 == 0:
            print(f"episode {e+1}/{n_episodes}", flush=True)
    out = {
        "cold_X": np.stack(cold_X).astype(np.float32),
        "cold_y": np.asarray(cold_y, dtype=np.int64),
        "warm_X": np.stack(warm_X).astype(np.float32),
        "warm_y": np.asarray(warm_y, dtype=np.int64),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["train", "metadev"], required=True)
    ap.add_argument("--n-episodes", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--known-set-sizes", default=None,
                    help="comma list, e.g. 2,3,4,6,8 (train-time bank sizes)")
    ap.add_argument("--evidence-mode", choices=["active", "fixed48"],
                    default="active")
    args = ap.parse_args()

    split = json.loads((ROOT / "outputs/iclr27_phase4w/meta_split/capacity.json").read_text())
    pool = split["meta_train_categories"] if args.split == "train" else split["meta_dev_categories"]
    sizes = ([int(x) for x in args.known_set_sizes.split(",")]
             if args.known_set_sizes else None)
    data = build_samples(args.n_episodes, args.seed, pool, args.device,
                         known_set_sizes=sizes,
                         evidence_mode=args.evidence_mode)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "samples.npz", **data)
    (out / "meta.json").write_text(json.dumps({
        "split": args.split, "pool": pool, "n_episodes": args.n_episodes,
        "seed": args.seed,
        "known_set_sizes": sizes or [4],
        "evidence_mode": args.evidence_mode,
        "cold_dim": int(data["cold_X"].shape[1]),
        "warm_dim": int(data["warm_X"].shape[1]),
        "cold_n": int(len(data["cold_y"])),
        "warm_n": int(len(data["warm_y"])),
        "cold_labels": {str(k): int(v) for k, v in
                        zip(*np.unique(data["cold_y"], return_counts=True))},
        "warm_labels": {str(k): int(v) for k, v in
                        zip(*np.unique(data["warm_y"], return_counts=True))},
    }, indent=2))
    print(json.dumps({
        "split": args.split, "cold_n": int(len(data["cold_y"])),
        "warm_n": int(len(data["warm_y"])),
        "cold_dim": int(data["cold_X"].shape[1]),
        "warm_dim": int(data["warm_X"].shape[1]),
        "cold_labels": {str(k): int(v) for k, v in
                        zip(*np.unique(data["cold_y"], return_counts=True))},
        "warm_labels": {str(k): int(v) for k, v in
                        zip(*np.unique(data["warm_y"], return_counts=True))},
    }, indent=2))


if __name__ == "__main__":
    main()
