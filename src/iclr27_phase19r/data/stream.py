"""Phase19R frozen DINOv2 stream and legal known-only metadata.

The source CSV is evaluator-side data.  Category values are retained in this
process only for loss-side episode construction and post-freeze scoring; model
inputs contain an episode-local role and boolean known mask, never a category
identifier.
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
SRC = ROOT / "data/iclr27_phase19r/sources"
OUT = ROOT / "outputs/iclr27_phase19r"
GEOM = [
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_box_stability_iou",
]


def _key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


class Phase19RData:
    """Memory-mapped feature source with fold-local known fitting."""

    def __init__(self, fold: int = 0, final: bool = False):
        self.fold = int(fold)
        self.final = bool(final)
        self.rows = list(csv.DictReader((SRC / "public_rows_corrected.csv").open(newline="")))
        self.supported_ids = sorted(int(x) for x in json.loads((SRC / "supported_known_ids.json").read_text()))
        self.supported_set = set(self.supported_ids)
        self.known_to_index = {c: i for i, c in enumerate(self.supported_ids)}
        manifest_path = OUT / "manifests/fold_manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"folds": []}
        self.fold_record = {"held_categories": [], "fit_videos": [], "validation_videos": []}
        if not final and self.manifest.get("folds"):
            self.fold_record = self.manifest["folds"][self.fold]
        self.held_categories = set(int(x) for x in self.fold_record.get("held_categories", []))

        z = np.load(SRC / "public_cls_roi.npz", mmap_mode="r")
        assert z["cls"].shape[0] == len(self.rows)
        cls = np.asarray(z["cls"], dtype=np.float32)
        roi = np.asarray(z["roi"], dtype=np.float32)
        raw = .8 * cls + .2 * roi
        raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-6)
        geom = np.asarray([[float(r[k]) for k in GEOM] for r in self.rows], np.float32)
        self.fit_rows = self._fit_rows()
        self.geom_mean = geom[self.fit_rows].mean(0) if len(self.fit_rows) else np.zeros(len(GEOM), np.float32)
        self.geom_std = np.maximum(geom[self.fit_rows].std(0), 1e-4) if len(self.fit_rows) else np.ones(len(GEOM), np.float32)
        self.raw = raw.astype(np.float32)
        self.geom = ((geom - self.geom_mean) / self.geom_std).astype(np.float32)

        self.track_rows: dict[str, list[int]] = defaultdict(list)
        self._prefix_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, float, int]] = {}
        self.track_video: dict[str, int] = {}
        self.track_category: dict[str, int] = {}
        self.track_role: dict[str, str] = {}
        for i, row in enumerate(self.rows):
            key = _key(row)
            self.track_rows[key].append(i)
            self.track_video[key] = int(row["video_id"])
            self.track_category[key] = int(row["gt_category_id_common"])
            self.track_role[key] = row.get("gt_role_common", "")
        for key, idx in self.track_rows.items():
            idx.sort(key=lambda i: (int(self.rows[i]["event_rank"]), i))

        # Supported-known tracklets are legal same-track positives.  All
        # category values remain evaluator/loss metadata, not feature fields.
        self.category_tracks: dict[int, list[str]] = defaultdict(list)
        self.category_videos: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, cat in self.track_category.items():
            if cat not in self.supported_set or self.track_role[key] != "supported_known":
                continue
            self.category_tracks[cat].append(key)
            self.category_videos[cat][self.track_video[key]].append(key)
        for cat in self.category_tracks:
            self.category_tracks[cat] = sorted(set(self.category_tracks[cat]))
            for video in self.category_videos[cat]:
                self.category_videos[cat][video] = sorted(set(self.category_videos[cat][video]))
        audit_path = OUT / "audit/eligible_category_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            self.eligible_categories = sorted(int(c) for c, x in audit["categories"].items() if x["eligible"])
        else:
            self.eligible_categories = sorted(c for c, by in self.category_videos.items() if len(by) >= 4)
        self.train_categories = sorted(c for c in self.supported_ids if c not in self.held_categories)
        self.eligible_train_categories = [c for c in self.eligible_categories if c in self.train_categories]
        self.eligible_held_categories = [c for c in self.eligible_categories if c in self.held_categories]
        self.known_prototypes, self.known_counts = self._build_known_prototypes()
        self.known_bias = np.zeros(len(self.supported_ids), np.float32)
        # A separate known stage may replace centroid prototypes with a frozen
        # normalized linear classifier.  The artifact is created before novel
        # policy training and is never optimized by RC-MS-OCD.
        known_stage = OUT / "checkpoints" / ("known_stage_final.npz" if self.final else f"known_stage_fold{self.fold}.npz")
        if known_stage.exists():
            kz = np.load(known_stage)
            if kz["prototypes"].shape == self.known_prototypes.shape:
                self.known_prototypes = np.asarray(kz["prototypes"], np.float32)
                self.known_counts = np.asarray(kz["counts"], np.int64)
                if "bias" in kz:
                    self.known_bias = np.asarray(kz["bias"], np.float32)
        self.active_known_mask = self.known_counts > 0
        self.trainer_observed_semantic_values = sorted(set(self.train_categories + [-1]))
        assert self.raw.shape == (len(self.rows), 768)
        assert set(self.trainer_observed_semantic_values) <= self.supported_set | {-1}

    def _fit_rows(self) -> np.ndarray:
        if self.final:
            videos = {int(r["video_id"]) for r in self.rows if int(r["gt_category_id_common"]) in self.supported_set}
        else:
            videos = set(int(v) for v in self.fold_record.get("fit_videos", self.fold_record.get("train_videos", [])))
        return np.asarray([i for i, r in enumerate(self.rows)
                           if int(r["video_id"]) in videos and int(r["gt_category_id_common"]) in self.supported_set], np.int64)

    @property
    def fit_videos(self) -> set[int]:
        if self.final:
            return {int(r["video_id"]) for r in self.rows if int(r["gt_category_id_common"]) in self.supported_set}
        return set(int(v) for v in self.fold_record.get("fit_videos", self.fold_record.get("train_videos", [])))

    @property
    def validation_videos(self) -> set[int]:
        return set(int(v) for v in self.fold_record.get("validation_videos", []))

    def _build_known_prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        proto = np.zeros((len(self.supported_ids), self.raw.shape[1]), np.float32)
        counts = np.zeros(len(self.supported_ids), np.int64)
        allowed_videos = self.fit_videos
        for key, cat in self.track_category.items():
            if cat not in self.supported_set or cat in self.held_categories or self.track_video[key] not in allowed_videos:
                continue
            idx = [i for i in self.track_rows[key]
                   if self.rows[i].get("gt_role_common") == "supported_known"
                   and self.rows[i].get("assigned") == "1" and float(self.rows[i].get("row_iou", 0.0)) >= .5]
            if not idx:
                idx = [i for i in self.track_rows[key] if self.rows[i].get("gt_role_common") == "supported_known"]
            if not idx:
                continue
            # Prefix endpoint is causal and stable for known-stage prototypes.
            w = np.asarray([self._quality_row(i) for i in idx], np.float32)
            v = np.average(self.raw[idx], axis=0, weights=np.maximum(w, .02))
            j = self.known_to_index[cat]
            proto[j] += v.astype(np.float32)
            counts[j] += 1
        for j in range(len(self.supported_ids)):
            if counts[j]:
                proto[j] /= float(counts[j])
                proto[j] /= max(float(np.linalg.norm(proto[j])), 1e-6)
        return proto, counts

    def _quality_row(self, i: int) -> float:
        r = self.rows[i]
        return float(np.clip(.62 * float(r["score"]) + .23 * float(r["causal_box_stability_iou"])
                             + .15 * min(1., math.log1p(int(r["causal_prefix_count"])) / math.log(5.)), 0., 1.))

    def prefix(self, track_key: str, position: int | None = None) -> tuple[np.ndarray, np.ndarray, float, int]:
        idx = self.track_rows[track_key]
        if position is None:
            position = len(idx) - 1
        position = max(0, min(int(position), len(idx) - 1))
        cached = self._prefix_cache.get((track_key, position))
        if cached is not None:
            return cached
        take = idx[:position + 1]
        w = np.asarray([self._quality_row(i) for i in take], np.float32)
        w = np.maximum(w, .02)
        raw = np.average(self.raw[take], axis=0, weights=w).astype(np.float32)
        raw /= max(float(np.linalg.norm(raw)), 1e-6)
        geom = np.average(self.geom[take], axis=0, weights=w).astype(np.float32)
        quality = float(np.clip(np.average([self._quality_row(i) for i in take], weights=w), 0., 1.))
        result = (raw, geom, quality, position)
        self._prefix_cache[(track_key, position)] = result
        return result

    def track_candidates(self, category: int, *, videos: set[int] | None = None) -> list[str]:
        keys = self.category_tracks.get(int(category), [])
        if videos is not None:
            keys = [k for k in keys if self.track_video[k] in videos]
        return list(keys)

    def random_track(self, rng: np.random.Generator, category: int, exclude_video: int | None = None,
                     allowed_videos: set[int] | None = None) -> str:
        choices = self.track_candidates(category, videos=allowed_videos)
        if exclude_video is not None:
            other = [k for k in choices if self.track_video[k] != exclude_video]
            if other:
                choices = other
        if not choices:
            raise ValueError(f"no legal track for category {category}")
        return choices[int(rng.integers(len(choices)))]

    def hard_negative_track(self, rng: np.random.Generator, query_key: str, excluded: set[int]) -> tuple[str, float]:
        q, _, _, _ = self.prefix(query_key)
        best: tuple[str, float] | None = None
        for cat in self.eligible_train_categories + self.eligible_held_categories:
            if cat in excluded:
                continue
            for key in self.category_tracks.get(cat, []):
                if self.track_video[key] == self.track_video[query_key]:
                    continue
                v, _, _, _ = self.prefix(key)
                score = float(q @ v)
                if best is None or score > best[1]:
                    best = (key, score)
        if best is None:
            raise ValueError("no cross-category hard negative available")
        return best

    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold, "final": self.final, "source_rows": len(self.rows),
            "fit_rows": int(len(self.fit_rows)), "tracklets": len(self.track_rows),
            "eligible_categories": self.eligible_categories,
            "held_categories": sorted(self.held_categories),
            "active_supported_known_ids": [self.supported_ids[i] for i, x in enumerate(self.active_known_mask) if x],
            "trainer_observed_semantic_values": self.trainer_observed_semantic_values,
            "true_novel_labels_in_model_input": False,
            "physical_id_used_as_feature": False,
        }

    def make_episode(self, rng: np.random.Generator, ladder: str = "L2"):
        from src.iclr27_phase19r.data.episodes import EpisodeFactory
        return EpisodeFactory(self, ladder=ladder).sample(rng)
