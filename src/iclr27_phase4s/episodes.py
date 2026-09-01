"""Pseudo-novel episodic benchmark construction for Phase 4S.

Legal universe: 48 train-supported known categories (2196 physical tracks,
<=8 frames DINOv2 768-d each). Meta-train classes build training episodes;
meta-dev classes build episodic dev. Test-novel GT is never touched.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.iclr27_phase4s.protocol import (
    load_frame_features,
    load_train_tracks,
    meta_split_classes,
)

OUT = Path("outputs/iclr27_phase4s/episodes")
MAX_OCC = 24  # upper bound on occurrences per episode (novel 9 + known 8 + FP <=7)


@dataclass
class EpisodeConfig:
    num_pseudo_known: int = 4
    num_pseudo_novel: int = 3
    tracks_per_novel: int = 3
    tracks_per_known: int = 2
    fp_ratio: float = 0.2
    min_commit_age: int = 2
    r_phys_floor: float = 0.4
    seed: int = 20260814


def build_universe(meta_classes: set[int]) -> dict[int, list[str]]:
    """category -> sorted physical track sample_ids, only meta split classes."""
    rows = load_train_tracks()
    by_cat: dict[int, list[str]] = {}
    for r in rows:
        c = int(r["category_id"])
        if c in meta_classes:
            by_cat.setdefault(c, []).append(r["sample_id"])
    for c in by_cat:
        by_cat[c].sort()
    return by_cat


def make_episode(
    by_cat: dict[int, list[str]],
    features: dict[str, np.ndarray],
    cfg: EpisodeConfig,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> dict:
    """One episode: shuffled per-track stream with causal memory labels.

    Occurrence schema:
      sid, category (int; -1 for FP), role ('known'/'novel'/'fp'),
      first (bool, novel first occurrence), frames (N x 768), r_phys (N,)

    Teacher slot bookkeeping is computed by the caller (train/eval) because
    it depends on stream order; here we only fix stream order + occurrence GT.
    """
    all_cats = sorted(by_cat)
    rng.shuffle(all_cats)
    novel_pool = [c for c in all_cats if len(by_cat[c]) >= 2]
    if len(novel_pool) < cfg.num_pseudo_novel:
        novel_pool = all_cats
    rng.shuffle(novel_pool)
    pseudo_novel = novel_pool[: cfg.num_pseudo_novel]
    known_pool = [c for c in all_cats if c not in pseudo_novel]
    rng.shuffle(known_pool)
    pseudo_known = known_pool[: cfg.num_pseudo_known]

    occurrences: list[dict] = []
    for c in pseudo_novel:
        ids = by_cat[c][:]
        rng.shuffle(ids)
        for i, sid in enumerate(ids[: cfg.tracks_per_novel]):
            occurrences.append({
                "sid": sid, "category": c, "role": "novel", "first": i == 0,
            })
    for c in pseudo_known:
        ids = by_cat[c][:]
        rng.shuffle(ids)
        for sid in ids[: cfg.tracks_per_known]:
            occurrences.append({
                "sid": sid, "category": c, "role": "known", "first": False,
            })

    n_fp = max(1, int(round(cfg.fp_ratio * len(occurrences))))
    fp_src = [sid for c in all_cats for sid in by_cat[c]]
    for _ in range(n_fp):
        sid = rng.choice(fp_src)
        occurrences.append({
            "sid": sid, "category": -1, "role": "fp", "first": False,
        })
    rng.shuffle(occurrences)
    # "first" must mean first-in-stream, not first-in-pre-shuffle order
    seen: set[int] = set()
    for occ in occurrences:
        if occ["role"] == "novel":
            occ["first"] = occ["category"] not in seen
            seen.add(occ["category"])

    out = []
    for occ in occurrences:
        src = features[occ["sid"]].astype(np.float32)
        n = len(src)
        if occ["role"] == "fp":
            # synthetic unreliable track: other-category features + noise
            frames = src + np_rng.normal(0.0, 0.25, size=src.shape).astype(np.float32)
            frames /= np.linalg.norm(frames, axis=1, keepdims=True) + 1e-12
            r_phys = np_rng.uniform(0.05, 0.3, size=n).astype(np.float32)
        else:
            frames = src
            r_phys = np_rng.uniform(0.75, 1.0, size=n).astype(np.float32)
        out.append({
            "sid": occ["sid"], "category": occ["category"], "role": occ["role"],
            "first": occ["first"], "frames": frames, "r_phys": r_phys,
        })
    return {
        "pseudo_known": sorted(pseudo_known),
        "pseudo_novel": sorted(pseudo_novel),
        "occurrences": out,
    }


def episode_to_batch(episode: dict, cfg: EpisodeConfig, teacher_slots: dict[int, int], max_len: int):
    """Build padded tensors + per-step teacher target codes for one episode."""
    n_occ = len(episode["occurrences"])
    feats = torch.zeros(MAX_OCC, max_len, 768)
    r_phys = torch.zeros(MAX_OCC, max_len)
    mask = torch.zeros(MAX_OCC, max_len, dtype=torch.bool)
    cat = torch.zeros(MAX_OCC, dtype=torch.long) - 1
    role_first = []
    for i, occ in enumerate(episode["occurrences"]):
        n = len(occ["frames"])
        feats[i, :n] = torch.from_numpy(occ["frames"])
        r_phys[i, :n] = torch.from_numpy(occ["r_phys"])
        mask[i, :n] = True
        if occ["role"] in ("novel", "known"):
            cat[i] = occ["category"]
        role_first.append((occ["role"], occ["first"]))
    while len(role_first) < MAX_OCC:
        role_first.append(("fp", False))
    return {
        "feats": feats, "r_phys": r_phys, "mask": mask,
        "cats": cat, "role_first": role_first,
        "pseudo_known": episode["pseudo_known"],
        "pseudo_novel": episode["pseudo_novel"],
    }


class EpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        by_cat: dict[int, list[str]],
        features: dict[str, np.ndarray],
        cfg: EpisodeConfig,
        n_episodes: int,
        max_len: int = 8,
        seed: int = 0,
    ):
        self.by_cat = by_cat
        self.features = features
        self.cfg = cfg
        self.n_episodes = n_episodes
        self.max_len = max_len
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n_episodes

    def __getitem__(self, idx):
        # deterministic per-index episode so dataloader workers are reproducible
        s = (self.seed * 10**6 + idx) % (2**32)
        rng = random.Random(s)
        np_rng = np.random.RandomState(s)
        ep = make_episode(self.by_cat, self.features, self.cfg, rng, np_rng)
        return episode_to_batch(ep, self.cfg, {}, self.max_len)


def collate_episodes(batch):
    out = {
        "feats": torch.stack([b["feats"] for b in batch]),
        "r_phys": torch.stack([b["r_phys"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "cats": torch.stack([b["cats"] for b in batch]),
        "role_first": [b["role_first"] for b in batch],
        "pseudo_known": [b["pseudo_known"] for b in batch],
        "pseudo_novel": [b["pseudo_novel"] for b in batch],
    }
    return out


def category_prototypes(features: dict[str, np.ndarray], by_cat: dict[int, list[str]]):
    """Mean prototype per category from the legal train-known universe."""
    protos = {}
    for c, ids in by_cat.items():
        arr = np.stack([features[s].mean(axis=0) for s in ids])
        m = arr.mean(axis=0)
        protos[c] = m / (np.linalg.norm(m) + 1e-12)
    return protos


def load_episodic_universe():
    meta_train, meta_dev = meta_split_classes()
    features = load_frame_features()
    by_train = build_universe(meta_train)
    by_dev = build_universe(meta_dev)
    return by_train, by_dev, features
