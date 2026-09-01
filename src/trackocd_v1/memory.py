#!/usr/bin/env python3
"""Seeded candidate-mature multi-prototype memory used by the architecture
bake-off. Known prototypes are pre-seeded as mature categories; novel
categories are created/attached with the same dual-threshold rules as the
Architecture 1.5 OCD-v2 memory. This is a controlled replacement so that
A3 (mean) and M3 (trajectory encoder) differ only in representation."""
from __future__ import annotations

import numpy as np


class SeededMultiPrototypeMemory:
    def __init__(
        self,
        known_protos,
        attach_thr=0.525,
        create_thr=0.375,
        new_proto_thr=0.475,
        max_proto=4,
        ema=0.25,
        maturity_tracks=2,
    ):
        self.attach_thr = attach_thr
        self.create_thr = create_thr
        self.new_proto_thr = new_proto_thr
        self.max_proto = max_proto
        self.ema = ema
        self.maturity_tracks = maturity_tracks
        self.next_id = 200000
        self.categories = {}
        for sem_id, p in known_protos.items():
            self.categories[int(sem_id)] = {
                "prototypes": [np.asarray(p, dtype=np.float32)],
                "track_count": 0,
                "videos": set(),
                "confidence": 1.0,
                "last_update": -1,
                "status": "mature",
                "semantic": int(sem_id),
            }

    def _cat_score(self, x, cid):
        return max(float(np.dot(x, p)) for p in self.categories[cid]["prototypes"])

    def predict_one(self, x, sample_id, stream_order, num_frames=1, video_id=None, score=1.0):
        best_sim, best_id = -1.0, None
        for cid in self.categories:
            s = self._cat_score(x, cid)
            if s > best_sim:
                best_sim, best_id = s, cid
        if best_sim >= self.attach_thr and best_id is not None:
            vid = best_id
            cat = self.categories[vid]
            idx = int(np.argmax([float(np.dot(x, p)) for p in cat["prototypes"]]))
            conf = best_sim
            w = self.ema * (0.5 + 0.5 * conf)
            cat["prototypes"][idx] = (1 - w) * cat["prototypes"][idx] + w * x
            cat["prototypes"][idx] /= np.linalg.norm(cat["prototypes"][idx]) + 1e-12
            cat["track_count"] += 1
            cat["confidence"] = 0.9 * cat["confidence"] + 0.1 * conf
            if video_id is not None:
                cat["videos"].add(video_id)
            cat["last_update"] = stream_order
            if (
                cat["status"] != "mature"
                and len(cat["prototypes"]) < self.max_proto
                and best_sim < self.new_proto_thr
                and (cat["track_count"] >= 2 or len(cat["videos"]) >= 2)
            ):
                cat["prototypes"].append(x.copy())
        elif best_sim <= self.create_thr or best_id is None:
            vid = self.next_id
            self.next_id += 1
            self.categories[vid] = {
                "prototypes": [x.copy()],
                "track_count": 1,
                "videos": {video_id} if video_id is not None else set(),
                "confidence": float(best_sim) if best_sim > 0 else 0.0,
                "last_update": stream_order,
                "status": "candidate",
                "semantic": None,
            }
        else:
            vid = best_id
            cat = self.categories[vid]
            idx = int(np.argmax([float(np.dot(x, p)) for p in cat["prototypes"]]))
            w = self.ema * 0.25
            cat["prototypes"][idx] = (1 - w) * cat["prototypes"][idx] + w * x
            cat["prototypes"][idx] /= np.linalg.norm(cat["prototypes"][idx]) + 1e-12
            cat["track_count"] += 1
            cat["confidence"] = 0.95 * cat["confidence"] + 0.05 * best_sim
            if video_id is not None:
                cat["videos"].add(video_id)
            cat["last_update"] = stream_order
        cat = self.categories[vid]
        if cat["status"] != "mature":
            if cat["track_count"] >= self.maturity_tracks and cat["confidence"] >= self.attach_thr:
                cat["status"] = "mature"
        return vid

    def memory_stats(self):
        return {
            "categories": len(self.categories),
            "mature": sum(1 for c in self.categories.values() if c["status"] == "mature"),
            "candidate": sum(1 for c in self.categories.values() if c["status"] == "candidate"),
            "prototypes": sum(len(c["prototypes"]) for c in self.categories.values()),
        }
