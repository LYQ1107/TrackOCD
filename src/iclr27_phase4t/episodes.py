"""Real tracker-induced pseudo-novel episodes for Phase 4T.

Occurrences are REAL Q1 physical tracklets (real score / prior_hits / age /
gap / fragmentation), with semantic supervision from the 48 supported-known
train categories only. Real FP tracklets are the unreliability examples.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from src.iclr27_phase4t.stream_data import build_tracklets


@dataclass
class RealEpisodeConfig:
    num_pseudo_known: int = 4
    num_pseudo_novel: int = 3
    tracklets_per_novel: int = 3
    tracklets_per_known: int = 2
    fp_per_episode: int = 4
    min_commit_age: int = 2
    r_phys_floor: float = 0.4
    max_len: int = 8
    seed: int = 20260814


class RealStreamStore:
    def __init__(self, rows: list[dict], feats: np.ndarray):
        self.rows = rows
        self.row_index = {id(r): i for i, r in enumerate(rows)}
        self.feats = feats
        self.tracklets = build_tracklets(rows)
        self.by_cat: dict[int, list] = {}
        for key, tl in self.tracklets.items():
            if tl["role"] == "known":
                self.by_cat.setdefault(tl["gt_category_id"], []).append(key)
        self.fp_tracklets = [k for k, tl in self.tracklets.items() if tl["role"] == "fp"]

    def tracklet_seq(self, key) -> tuple[np.ndarray, np.ndarray]:
        rows = self.tracklets[key]["rows"]
        idx = [self.row_index[id(r)] for r in rows]
        z = np.stack([self.feats[i] for i in idx]).astype(np.float32)
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        q = np.stack([r["q_phys"] for r in rows]).astype(np.float32)
        return z, q


def make_real_episode(store: RealStreamStore, cfg: RealEpisodeConfig,
                      rng: random.Random) -> dict:
    cats = sorted(store.by_cat)
    rng.shuffle(cats)
    novel_pool = [c for c in cats if len(store.by_cat[c]) >= 2]
    rng.shuffle(novel_pool)
    pseudo_novel = novel_pool[: cfg.num_pseudo_novel]
    known_pool = [c for c in cats if c not in pseudo_novel]
    rng.shuffle(known_pool)
    pseudo_known = known_pool[: cfg.num_pseudo_known]

    occurrences = []
    for c in pseudo_novel:
        keys = store.by_cat[c][:]
        rng.shuffle(keys)
        # prefer cross-video positives: sort so distinct videos come first
        vids = {}
        for k in keys:
            vids.setdefault(k[0], []).append(k)
        picked = []
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
    if not fp_pool:
        fp_pool = store.fp_tracklets
    rng.shuffle(fp_pool)
    for k in fp_pool[: cfg.fp_per_episode]:
        occurrences.append({"key": k, "role": "fp", "category": -1})
    rng.shuffle(occurrences)
    # "first" = first-in-stream for each pseudo-novel category
    seen = set()
    for occ in occurrences:
        if occ["role"] == "novel":
            occ["first"] = occ["category"] not in seen
            seen.add(occ["category"])
    return {"pseudo_known": pseudo_known, "pseudo_novel": pseudo_novel,
            "occurrences": occurrences}


MAX_OCC = 24


def real_episode_batch(store: RealStreamStore, episode: dict, cfg: RealEpisodeConfig):
    n_occ = len(episode["occurrences"])
    feats = torch.zeros(MAX_OCC, cfg.max_len, 768)
    qphys = torch.zeros(MAX_OCC, cfg.max_len, 6)
    rphys = torch.zeros(MAX_OCC, cfg.max_len)
    mask = torch.zeros(MAX_OCC, cfg.max_len, dtype=torch.bool)
    cats = torch.zeros(MAX_OCC, dtype=torch.long) - 1
    role_first = []
    for i, occ in enumerate(episode["occurrences"]):
        z, q = store.tracklet_seq(occ["key"])
        n = min(len(z), cfg.max_len)
        feats[i, :n] = torch.from_numpy(z[:n])
        qphys[i, :n] = torch.from_numpy(q[:n])
        rphys[i, :n] = torch.from_numpy(q[:n, 0]).clamp(0.05, 0.95)
        mask[i, :n] = True
        if occ["role"] in ("novel", "known"):
            cats[i] = occ["category"]
        role_first.append((occ["role"], occ.get("first", False)))
    while len(role_first) < MAX_OCC:
        role_first.append(("fp", False))
    return {"feats": feats, "qphys": qphys, "r_phys": rphys, "mask": mask, "cats": cats,
            "role_first": role_first,
            "pseudo_known": episode["pseudo_known"],
            "pseudo_novel": episode["pseudo_novel"]}


class RealEpisodeDataset(torch.utils.data.Dataset):
    def __init__(self, store: RealStreamStore, cfg: RealEpisodeConfig,
                 n_episodes: int, seed: int = 0):
        self.store = store
        self.cfg = cfg
        self.n_episodes = n_episodes
        self.seed = seed

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        rng = random.Random((self.seed * 10**6 + idx) % (2**32))
        ep = make_real_episode(self.store, self.cfg, rng)
        return real_episode_batch(self.store, ep, self.cfg)


def collate_real(batch):
    return {
        "feats": torch.stack([b["feats"] for b in batch]),
        "qphys": torch.stack([b["qphys"] for b in batch]),
        "r_phys": torch.stack([b["r_phys"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "cats": torch.stack([b["cats"] for b in batch]),
        "role_first": [b["role_first"] for b in batch],
        "pseudo_known": [b["pseudo_known"] for b in batch],
        "pseudo_novel": [b["pseudo_novel"] for b in batch],
    }
