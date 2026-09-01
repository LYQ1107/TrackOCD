"""Causal known-novel bi-memory and action statistics."""
from __future__ import annotations

import numpy as np
import torch


class BiMemory:
    def __init__(self, known_protos, known_radii=None, novel_update_rate=0.2):
        self.known = {int(k): np.asarray(v, dtype=np.float32) for k, v in known_protos.items()}
        self.known_radii = known_radii or {}
        self.novel = {}
        self.novel_counts = {}
        self.novel_radii = {}
        self.novel_update_rate = novel_update_rate
        self.next_id = 100000

    def _norm(self, x):
        x = np.asarray(x, dtype=np.float32)
        n = np.linalg.norm(x)
        return x / n if n > 0 else x

    def stats(self, z: np.ndarray, reliability: float, track_len: int,
              known_id=None, novel_id=None) -> list[float]:
        best_k, second_k = -1.0, -1.0
        best_k_id = None
        for cid, p in self.known.items():
            s = float(np.dot(z, p))
            if s > best_k:
                second_k = best_k
                best_k, best_k_id = s, cid
            elif s > second_k:
                second_k = s
        r_k = float(self.known_radii.get(known_id if known_id is not None else best_k_id, 1.0))
        best_n, second_n = -1.0, -1.0
        best_n_id = None
        for cid, c in self.novel.items():
            p = c["proto"]
            s = float(np.dot(z, p))
            if s > best_n:
                second_n = best_n
                best_n, best_n_id = s, cid
            elif s > second_n:
                second_n = s
        margin_k = best_k - second_k if second_k > -1 else 0.0
        margin_n = best_n - second_n if second_n > -1 else 0.0
        dist_k = (1.0 - best_k) / max(r_k, 1e-6)
        r_n = float(self.novel_radii.get(novel_id if novel_id is not None else best_n_id, 0.3))
        dist_n = (1.0 - best_n) / max(r_n, 1e-6)
        return [
            best_k, second_k, margin_k, dist_k,
            best_n, second_n, margin_n, dist_n,
            reliability, float(track_len) / 40.0, float(len(self.novel)) / 300.0,
        ]

    def known_id(self, z: np.ndarray) -> tuple[int, float]:
        best_id, best_s = None, -1.0
        for cid, p in self.known.items():
            s = float(np.dot(z, p))
            if s > best_s:
                best_s, best_id = s, cid
        return best_id, best_s

    def existing_novel(self, z: np.ndarray) -> tuple[int, float]:
        best_id, best_s = None, -1.0
        for cid, c in self.novel.items():
            s = float(np.dot(z, c["proto"]))
            if s > best_s:
                best_s, best_id = s, cid
        return best_id, best_s

    def create_novel(self, z: np.ndarray) -> int:
        vid = self.next_id
        self.next_id += 1
        self.novel[vid] = {"proto": self._norm(z)}
        self.novel_counts[vid] = 1
        self.novel_radii[vid] = 0.3
        return vid

    def update_novel(self, vid: int, z: np.ndarray):
        c = self.novel[vid]
        n = self.novel_counts.get(vid, 1)
        w = self.novel_update_rate
        c["proto"] = self._norm((1 - w) * c["proto"] + w * z)
        self.novel_counts[vid] = n + 1


def stats_to_tensor(stats: list[float], device="cpu") -> torch.Tensor:
    return torch.as_tensor([stats], dtype=torch.float32, device=device)
