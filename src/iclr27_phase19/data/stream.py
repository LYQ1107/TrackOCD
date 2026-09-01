"""Phase19 row-aligned, memory-mapped feature and episode source.

The loader keeps evaluator-only category metadata separate from the tensors
passed to the model.  Non-supported semantic values are mapped to -1 before
the model-facing batch is constructed.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "data/iclr27_phase19/sources"
OUT = ROOT / "outputs/iclr27_phase19"
GEOM = [
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_box_stability_iou",
]


class Phase19Data:
    def __init__(self, fold: int = 0, final: bool = False):
        self.fold = int(fold)
        self.final = bool(final)
        self.rows = list(csv.DictReader((SRC / "public_rows_corrected.csv").open(newline="")))
        self.supported_ids = sorted(int(x) for x in json.loads((SRC / "supported_known_ids.json").read_text()))
        self.supported_set = set(self.supported_ids)
        self.known_to_index = {c: i for i, c in enumerate(self.supported_ids)}
        fm = json.loads((OUT / "manifests/fold_manifest.json").read_text())
        self.fold_record = {"held_categories": []} if final else fm["folds"][self.fold]
        self.held_categories = set(int(x) for x in self.fold_record.get("held_categories", []))
        self.train_categories = set(self.supported_ids) - self.held_categories
        z = np.load(SRC / "public_cls_roi.npz", mmap_mode="r")
        assert z["cls"].shape[0] == len(self.rows) == 43423
        cls = np.asarray(z["cls"], dtype=np.float32)
        roi = np.asarray(z["roi"], dtype=np.float32)
        raw = .65 * cls + .35 * roi
        raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-6)
        geom = np.asarray([[float(r[k]) for k in GEOM] for r in self.rows], np.float32)
        self.fit_rows = self._fit_rows()
        self.geom_mean = geom[self.fit_rows].mean(0)
        self.geom_std = np.maximum(geom[self.fit_rows].std(0), 1e-4)
        self.raw = raw.astype(np.float32)
        self.geom = ((geom - self.geom_mean) / self.geom_std).astype(np.float32)
        self.track_rows: dict[str, list[int]] = defaultdict(list)
        self.track_video: dict[str, int] = {}
        self.track_cat_eval: dict[str, int] = {}
        self.track_cat_model: dict[str, int] = {}
        self.track_role: dict[str, str] = {}
        for i, r in enumerate(self.rows):
            key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"
            self.track_rows[key].append(i)
            self.track_video[key] = int(r["video_id"])
            c = int(r["gt_category_id_common"])
            self.track_cat_eval[key] = c
            self.track_cat_model[key] = c if c in self.train_categories else -1
            self.track_role[key] = r["gt_role_common"]
        for k in self.track_rows:
            self.track_rows[k].sort(key=lambda i: (int(self.rows[i]["event_rank"]), i))
            if self.track_cat_eval[k] not in self.supported_set:
                self.track_cat_model[k] = -1
        self.categories_to_tracks: dict[int, list[str]] = defaultdict(list)
        self.category_videos: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for k, c in self.track_cat_model.items():
            if c < 0 or self.track_video[k] not in self.train_videos:
                continue
            self.categories_to_tracks[c].append(k)
            self.category_videos[c][self.track_video[k]].append(k)
        for c in self.categories_to_tracks:
            self.categories_to_tracks[c].sort()
            for v in self.category_videos[c]:
                self.category_videos[c][v].sort()
        self.episode_categories = sorted(c for c, d in self.category_videos.items() if len(d) >= 2)
        self.trainer_observed_semantic_values = sorted(set([r for r in self.track_cat_model.values() if r >= 0] + [-1]))
        assert set(self.trainer_observed_semantic_values) <= self.supported_set | {-1}
        assert self.raw.shape == (43423, 768)

    def _fit_rows(self) -> np.ndarray:
        if self.final:
            # Full final training uses only rows in videos with at least one
            # supported-known track, but never reads a true-novel semantic value.
            videos = {int(r["video_id"]) for r in self.rows if int(r["gt_category_id_common"]) in self.supported_set}
            return np.asarray([i for i, r in enumerate(self.rows) if int(r["video_id"]) in videos], np.int64)
        vids = set(self.fold_record["train_videos"])
        return np.asarray([i for i, r in enumerate(self.rows) if int(r["video_id"]) in vids], np.int64)

    @property
    def train_videos(self) -> set[int]:
        if self.final:
            return {int(r["video_id"]) for r in self.rows if int(r["gt_category_id_common"]) in self.supported_set}
        return set(int(x) for x in self.fold_record["train_videos"])

    def prefix(self, track_key: str, position: int | None = None) -> tuple[np.ndarray, np.ndarray, float, int]:
        idx = self.track_rows[track_key]
        if position is None:
            position = len(idx) - 1
        position = max(0, min(int(position), len(idx) - 1))
        take = idx[:position + 1]
        # Causal quality weighting only uses rows in the observed prefix.
        w = np.asarray([
            .62 * float(self.rows[i]["score"])
            + .23 * float(self.rows[i]["causal_box_stability_iou"])
            + .15 * min(1.0, math.log1p(int(self.rows[i]["causal_prefix_count"])) / math.log(5.0))
            for i in take
        ], np.float32)
        w = np.maximum(w, .02)
        raw = np.average(self.raw[take], axis=0, weights=w).astype(np.float32)
        raw /= max(float(np.linalg.norm(raw)), 1e-6)
        geom = np.average(self.geom[take], axis=0, weights=w).astype(np.float32)
        quality = float(np.clip(np.average([
            .65 * float(self.rows[i]["score"]) + .35 * float(self.rows[i]["causal_box_stability_iou"])
            for i in take
        ], weights=w), 0., 1.))
        return raw, geom, quality, position

    def random_track(self, rng: np.random.Generator, category: int | None = None,
                     exclude_video: int | None = None) -> str:
        if category is None:
            category = int(rng.choice(self.episode_categories))
        choices = [k for k in self.categories_to_tracks[int(category)]
                   if exclude_video is None or self.track_video[k] != exclude_video]
        if not choices:
            choices = self.categories_to_tracks[int(category)]
        return choices[int(rng.integers(len(choices)))]

    def make_episode(self, rng: np.random.Generator, ladder: str = "L0") -> list[dict[str, Any]]:
        # All category ids are metadata used to construct an episode-local
        # target.  Only opaque action indices reach the model.
        cat = int(rng.choice(self.episode_categories))
        source = self.random_track(rng, cat)
        target = self.random_track(rng, cat, exclude_video=self.track_video[source])
        known = int(rng.choice(self.episode_categories))
        known_track = self.random_track(rng, known)
        def choose_pos(k: str) -> int:
            n = len(self.track_rows[k])
            if ladder == "L0":
                return max(0, n - 1)
            if ladder == "L1":
                return int(rng.integers(max(1, n // 2), n))
            return int(rng.integers(0, n))
        out = []
        for k, pseudo, visible in [(source, True, False), (target, True, False), (known_track, False, True)]:
            pos = choose_pos(k)
            raw, geom, quality, pos = self.prefix(k, pos)
            out.append({"raw": raw, "geom": geom, "quality": quality, "category": int(self.track_cat_model[k]),
                        "pseudo": pseudo, "visible": visible, "track_key": k, "video": self.track_video[k],
                        "position": pos})
        return out

    def known_prototypes(self) -> np.ndarray:
        proto = np.zeros((len(self.supported_ids), 768), np.float32)
        counts = np.zeros(len(self.supported_ids), np.float32)
        for k, c in self.track_cat_model.items():
            if c < 0 or c not in self.known_to_index or self.track_video[k] not in self.train_videos:
                continue
            raw, _, _, _ = self.prefix(k)
            j = self.known_to_index[c]; proto[j] += raw; counts[j] += 1
        proto /= np.maximum(counts[:, None], 1.)
        proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-6)
        return proto.astype(np.float32)

    def summary(self) -> dict[str, Any]:
        return {"fold": self.fold, "final": self.final, "source_rows": len(self.rows),
                "fit_rows": int(len(self.fit_rows)), "fit_tracklets": len(self.categories_to_tracks),
                "episode_categories": self.episode_categories,
                "held_categories": sorted(self.held_categories),
                "trainer_observed_semantic_values": self.trainer_observed_semantic_values,
                "true_novel_labels_in_model_input": False}
