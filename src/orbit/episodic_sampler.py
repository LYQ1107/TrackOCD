"""Episodic pseudo-novel sampling for ORBIT training."""
from __future__ import annotations

import random

import numpy as np

from src.orbit.protocol import load_frame_features, load_train_labels, meta_classes


class EpisodicSampler:
    def __init__(self, seed: int = 1027, num_known=20, support_per_class=4,
                 query_per_class=4, max_frames=8):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.num_known = num_known
        self.support_per_class = support_per_class
        self.query_per_class = query_per_class
        self.max_frames = max_frames
        self.frames = load_frame_features("train_known_mean")
        self.labels = load_train_labels()
        self.meta_train = meta_classes("meta_train_classes")
        self.by_class = {}
        for sid, c in self.labels.items():
            if c in self.meta_train and sid in self.frames:
                self.by_class.setdefault(c, []).append(sid)
        for c in self.by_class:
            self.by_class[c].sort()

    def episode(self):
        classes = sorted(self.by_class)
        self.rng.shuffle(classes)
        known_classes = classes[: self.num_known]
        pseudo_novel_classes = classes[self.num_known :]
        support = {}
        query = []
        for c in known_classes:
            ids = self.by_class[c]
            self.rng.shuffle(ids)
            support[c] = ids[: self.support_per_class]
            chosen = ids[self.support_per_class : self.support_per_class + self.query_per_class]
            for sid in chosen:
                query.append({"sample_id": sid, "label": c, "known": True, "first": False})
        first_seen = set()
        for c in pseudo_novel_classes:
            ids = self.by_class[c]
            self.rng.shuffle(ids)
            chosen = ids[: self.query_per_class]
            for sid in chosen:
                first = c not in first_seen
                first_seen.add(c)
                query.append({"sample_id": sid, "label": c, "known": False, "first": first})
        self.rng.shuffle(query)
        return {
            "known_classes": known_classes,
            "pseudo_novel_classes": pseudo_novel_classes,
            "support": support,
            "query": query,
        }

    def frames_for(self, sid: str) -> np.ndarray:
        arr = self.frames[sid][: self.max_frames]
        return arr
