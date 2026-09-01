"""Deterministic full-stream plus balanced DSTM episode construction."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs/iclr27_phase18"
GEOMETRY_FIELDS = [
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm",
    "causal_prefix_age_norm", "causal_box_stability_iou",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


class FoldData:
    def __init__(self, fold_id: int, config: dict[str, Any]):
        self.fold_id = fold_id
        self.config = config
        with (ROOT / "data/iclr27_phase18/sources/public_rows_corrected.csv").open(newline="") as f:
            self.rows = list(csv.DictReader(f))
        manifest = list(csv.DictReader((OUT / "manifests/row_aligned_tracklet_manifest.csv").open(newline="")))
        assert len(self.rows) == len(manifest) == 43423
        z = np.load(ROOT / "data/iclr27_phase18/sources/public_dinov2_cls_roi.npz", mmap_mode="r")
        order = np.asarray([int(x["dinov2_index"]) for x in manifest], np.int64)
        cls = np.asarray(z["cls"][order], np.float32)
        roi = np.asarray(z["roi"][order], np.float32)
        self.track_manifest = {x["tracklet_key"]: x for x in load_jsonl(OUT / "manifests/tracklet_manifest.jsonl")}
        fold_manifest = json.loads((OUT / "manifests/fold_manifest.json").read_text())
        self.fold = fold_manifest["folds"][fold_id]
        self.held_cats = set(self.fold["held_categories"])
        self.cal_cats = set(self.fold["nested_calibration_categories"])
        self.held_videos = set(self.fold["strict_excluded_videos"])
        self.known_ids = [int(x) for x in json.loads((ROOT / "data/iclr27_phase18/sources/supported_known_ids.json").read_text())]
        self.known_to_index = {c: i for i, c in enumerate(self.known_ids)}
        geom = np.asarray([[float(r[x]) for x in GEOMETRY_FIELDS] for r in self.rows], np.float32)
        for i, r in enumerate(self.rows):
            r["idx"] = i; r["video_i"] = int(r["video_id"]); r["track_i"] = int(r["track_id"])
            r["cat_i"] = int(r["gt_category_id_common"]); r["event_i"] = int(r["event_rank"])
            r["reliable"] = r["assigned"] == "1" and float(r["row_iou"]) >= .5
        self.fit_indices = np.asarray([
            i for i, r in enumerate(self.rows)
            if r["role17"] in {"known_bank", "novel_correspondence_train"}
            and r["cat_i"] not in self.held_cats | self.cal_cats
            and r["video_i"] not in self.held_videos
        ], np.int64)
        assert len(self.fit_indices) == int(self.fold["fit_row_count"])
        self.geom_mean = geom[self.fit_indices].mean(0)
        self.geom_std = np.maximum(geom[self.fit_indices].std(0), 1e-4)
        geom = (geom - self.geom_mean) / self.geom_std
        # Float16 storage cuts per-worker RSS and PCIe traffic; tensors are cast
        # to float32 before the BF16 autocast region.
        self.row_input = np.concatenate([cls, roi, geom], axis=1).astype(np.float16)
        self.input_dim = self.row_input.shape[1]
        self.reliability = np.asarray([r["reliable"] for r in self.rows], bool)
        self.track_indices: dict[str, list[int]] = defaultdict(list)
        self.row_track: dict[int, str] = {}
        for i in self.fit_indices:
            r = self.rows[int(i)]; key = f"v{r['video_i']}:p{r['track_i']}"
            self.track_indices[key].append(int(i)); self.row_track[int(i)] = key
        for key in self.track_indices:
            self.track_indices[key].sort(key=lambda i: (self.rows[i]["event_i"], i))
        self.row_position = {i: p for key, values in self.track_indices.items() for p, i in enumerate(values)}
        self.track_info = self._track_info()
        self.tracks_by_cat_video: dict[int, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for key, item in self.track_info.items():
            if item["label"] >= 0 and item["reliable_positions"]:
                self.tracks_by_cat_video[item["label"]][item["video"]].append(key)
        for videos in self.tracks_by_cat_video.values():
            for v in videos:
                videos[v].sort()
        self.crossvideo_cats = sorted(c for c, v in self.tracks_by_cat_video.items() if len(v) >= 2)
        self.labeled_tracks = sorted(k for k, x in self.track_info.items() if x["label"] >= 0 and x["reliable_positions"])
        self.known_rows = [i for i in self.fit_indices if self.rows[int(i)]["gt_role_common"] == "supported_known" and self.reliability[int(i)]]
        self.unreliable_rows = [i for i in self.fit_indices if not self.reliability[int(i)]]
        self.reliable_rows = [i for i in self.fit_indices if self.reliability[int(i)] and self.rows[int(i)]["gt_role_common"] in {"novel", "supported_known"}]
        self.existing_targets = [k for k, x in self.track_info.items() if x["label"] in self.crossvideo_cats and x["reliable_positions"]]
        self.merge_targets = [k for k in self.existing_targets if self.track_info[k]["first_reliable"] > 0]
        self.state_summary = {k: self._summary(v) for k, v in self.track_indices.items()}
        self.known_class_weights = self._known_weights()
        pos = int(self.reliability[self.fit_indices].sum()); neg = len(self.fit_indices) - pos
        self.reliability_pos_weight = min(20.0, neg / max(pos, 1))
        assert self.crossvideo_cats and self.existing_targets and self.merge_targets

    def _track_info(self) -> dict[str, dict[str, Any]]:
        result = {}
        for key, indices in self.track_indices.items():
            reliable_labels = Counter(
                self.rows[i]["cat_i"] for i in indices
                if self.reliability[i] and self.rows[i]["gt_role_common"] in {"novel", "supported_known"}
            )
            assert len(reliable_labels) <= 1, (key, reliable_labels)
            label = next(iter(reliable_labels)) if reliable_labels else -1
            reliable_positions = [p for p, i in enumerate(indices) if self.reliability[i] and self.rows[i]["cat_i"] == label]
            role = next((self.rows[i]["gt_role_common"] for i in indices if self.reliability[i] and self.rows[i]["cat_i"] == label), "fp")
            result[key] = {
                "key": key, "indices": indices, "video": self.rows[indices[0]]["video_i"],
                "label": label, "role": role, "reliable_positions": reliable_positions,
                "first_reliable": reliable_positions[0] if reliable_positions else -1,
            }
        return result

    def _summary(self, indices: list[int]) -> np.ndarray:
        weights = np.asarray([
            .62 * float(self.rows[i]["score"])
            + .23 * float(self.rows[i]["causal_box_stability_iou"])
            + .15 * min(1.0, math.log1p(int(self.rows[i]["causal_prefix_count"])) / math.log(5.0))
            for i in indices
        ], np.float32)
        return np.average(self.row_input[indices].astype(np.float32), axis=0, weights=np.maximum(weights, .02)).astype(np.float16)

    def _known_weights(self) -> np.ndarray:
        counts = Counter(self.rows[int(i)]["cat_i"] for i in self.known_rows)
        w = np.ones(len(self.known_ids), np.float32)
        present = []
        for c, idx in self.known_to_index.items():
            if counts[c]:
                w[idx] = 1.0 / math.sqrt(counts[c]); present.append(w[idx])
        scale = 1.0 / np.mean(present) if present else 1.0
        for c, idx in self.known_to_index.items():
            w[idx] = w[idx] * scale if counts[c] else 1.0
        return w

    def sequence(self, track_key: str, position: int) -> list[int]:
        values = self.track_indices[track_key][: position + 1]
        return values[-int(self.config["model"]["max_causal_sequence_rows"]):]

    def _correct_source(self, target_key: str, rng: np.random.Generator) -> str:
        info = self.track_info[target_key]; videos = self.tracks_by_cat_video[info["label"]]
        choices = [k for v, keys in videos.items() if v != info["video"] for k in keys]
        assert choices
        return choices[int(rng.integers(len(choices)))]

    def _distractors(self, target_key: str, count: int, rng: np.random.Generator) -> list[str]:
        info = self.track_info[target_key]
        choices = [k for k in self.labeled_tracks
                   if self.track_info[k]["label"] != info["label"]
                   and self.track_info[k]["video"] != info["video"]]
        if not choices:
            choices = [k for k in self.labeled_tracks if self.track_info[k]["label"] != info["label"]]
        if not choices or count <= 0:
            return []
        replace = len(choices) < count
        picked = rng.choice(len(choices), size=count, replace=replace)
        return [choices[int(i)] for i in np.atleast_1d(picked)]

    def _episode(self, target_key: str, position: int, action: str,
                 rng: np.random.Generator, merge: bool = False,
                 deterministic_known: bool = False) -> dict[str, Any]:
        info = self.track_info[target_key]
        max_states = int(self.config["model"]["max_training_state_candidates"])
        state_count = int(rng.integers(1, max_states + 1)) if action != "DEFER" else int(rng.integers(0, max_states + 1))
        candidates: list[str] = []
        correct = None
        if action == "EXISTING":
            correct = self._correct_source(target_key, rng); candidates.append(correct)
            candidates.extend(self._distractors(target_key, state_count - 1, rng))
        else:
            candidates.extend(self._distractors(target_key, state_count, rng))
            if action == "DEFER" and info["label"] in self.crossvideo_cats and state_count and rng.random() < .5:
                candidates[0] = self._correct_source(target_key, rng)
        if candidates:
            order = rng.permutation(len(candidates)); candidates = [candidates[int(i)] for i in order]
        known_mask = np.ones(len(self.known_ids), bool)
        if not deterministic_known and action in {"EXISTING", "NEW", "DEFER"} and info["label"] in self.known_to_index:
            known_mask[self.known_to_index[info["label"]]] = False
        if action == "KNOWN":
            label = self.known_to_index[info["label"]]; known_aux = label
        elif action == "EXISTING":
            label = len(self.known_ids) + candidates.index(correct); known_aux = -1
        elif action == "NEW":
            label = len(self.known_ids) + max_states; known_aux = -1
        else:
            label = len(self.known_ids) + max_states + 1; known_aux = -1
        pre = None
        if merge:
            assert info["first_reliable"] > 0
            pre_pos = int(rng.integers(0, info["first_reliable"]))
            pre = self.sequence(target_key, pre_pos)
        seq = self.sequence(target_key, position)
        current = seq[-1]
        temporal = (len(seq) >= 2 and self.reliability[seq[-1]] and self.reliability[seq[-2]]
                    and self.rows[seq[-1]]["cat_i"] == self.rows[seq[-2]]["cat_i"])
        return {
            "query_indices": seq, "pre_indices": pre, "candidate_tracks": candidates,
            "known_mask": known_mask, "label": label, "known_aux": known_aux,
            "reliability": float(self.reliability[current]), "metric_slot": candidates.index(correct) if correct else -1,
            "merge": bool(merge), "temporal": bool(temporal), "action_name": action,
        }

    def deterministic_example(self, row_index: int, rng: np.random.Generator,
                              variant: str) -> dict[str, Any]:
        key = self.row_track[row_index]; pos = self.row_position[row_index]
        info = self.track_info[key]; r = self.rows[row_index]
        if variant == "b3":
            if r["gt_role_common"] == "supported_known" and r["assigned"] == "1" and info["label"] in self.known_to_index:
                action = "KNOWN"
            elif self.reliability[row_index] and info["label"] in self.crossvideo_cats:
                action = "EXISTING"
            else:
                action = "NEW"
        elif not self.reliability[row_index]:
            action = "DEFER"
        elif r["gt_role_common"] == "supported_known" and info["label"] in self.known_to_index:
            action = "KNOWN"
        elif info["label"] in self.crossvideo_cats:
            action = "EXISTING"
        else:
            action = "NEW"
        return self._episode(key, pos, action, rng, deterministic_known=True)

    def balanced_example(self, kind: str, rng: np.random.Generator,
                         variant: str) -> dict[str, Any]:
        if variant == "b3" and kind in {"DEFER", "MERGE"}:
            kind = "NEW"
        if kind == "KNOWN":
            i = int(self.known_rows[int(rng.integers(len(self.known_rows)))])
            return self._episode(self.row_track[i], self.row_position[i], "KNOWN", rng)
        if kind == "DEFER":
            i = int(self.unreliable_rows[int(rng.integers(len(self.unreliable_rows)))])
            return self._episode(self.row_track[i], self.row_position[i], "DEFER", rng)
        if kind in {"EXISTING", "MERGE"}:
            pool = self.merge_targets if kind == "MERGE" else self.existing_targets
            key = pool[int(rng.integers(len(pool)))]; info = self.track_info[key]
            pos = int(info["reliable_positions"][int(rng.integers(len(info["reliable_positions"])))])
            return self._episode(key, pos, "EXISTING", rng, merge=kind == "MERGE")
        i = int(self.reliable_rows[int(rng.integers(len(self.reliable_rows)))])
        return self._episode(self.row_track[i], self.row_position[i], "NEW", rng)

    def build_batch(self, step: int, seed: int, variant: str) -> dict[str, np.ndarray | list[str]]:
        rng = np.random.default_rng(seed * 1_000_003 + step)
        deterministic_n = int(self.config["training"]["deterministic_population_examples_per_update"])
        balanced_n = int(self.config["training"]["balanced_episode_examples_per_update"])
        start = ((step - 1) * deterministic_n) % len(self.fit_indices)
        positions = [(start + j) % len(self.fit_indices) for j in range(deterministic_n)]
        examples = [self.deterministic_example(int(self.fit_indices[p]), rng, variant) for p in positions]
        cycle = ["DEFER", "KNOWN", "EXISTING", "NEW", "MERGE", "EXISTING"]
        for j in range(balanced_n):
            examples.append(self.balanced_example(cycle[(step + j) % len(cycle)], rng, variant))
        return self.collate(examples)

    def collate(self, examples: list[dict[str, Any]]) -> dict[str, np.ndarray | list[str]]:
        b = len(examples); max_len = int(self.config["model"]["max_causal_sequence_rows"])
        max_states = int(self.config["model"]["max_training_state_candidates"])
        q = np.zeros((b, max_len, self.input_dim), np.float16)
        pre = np.zeros_like(q)
        lengths = np.ones(b, np.int64); pre_lengths = np.ones(b, np.int64)
        states = np.zeros((b, max_states, self.input_dim), np.float16)
        state_mask = np.zeros((b, max_states), bool)
        known_mask = np.ones((b, len(self.known_ids)), bool)
        labels = np.zeros(b, np.int64); known_aux = np.full(b, -1, np.int64)
        reliability = np.zeros(b, np.float32); metric_slot = np.full(b, -1, np.int64)
        merge = np.zeros(b, bool); temporal = np.zeros(b, bool); actions = []
        for j, ex in enumerate(examples):
            idx = ex["query_indices"][-max_len:]; q[j, :len(idx)] = self.row_input[idx]; lengths[j] = len(idx)
            if ex["pre_indices"]:
                pi = ex["pre_indices"][-max_len:]; pre[j, :len(pi)] = self.row_input[pi]; pre_lengths[j] = len(pi)
            else:
                pre[j, 0] = self.row_input[idx[0]]
            for k, track in enumerate(ex["candidate_tracks"][:max_states]):
                states[j, k] = self.state_summary[track]; state_mask[j, k] = True
            known_mask[j] = ex["known_mask"]; labels[j] = ex["label"]
            known_aux[j] = ex["known_aux"]; reliability[j] = ex["reliability"]
            metric_slot[j] = ex["metric_slot"]; merge[j] = ex["merge"]
            temporal[j] = ex["temporal"]; actions.append(ex["action_name"])
        return {
            "query": q, "lengths": lengths, "pre": pre, "pre_lengths": pre_lengths,
            "states": states, "state_mask": state_mask, "known_mask": known_mask,
            "labels": labels, "known_aux": known_aux, "reliability": reliability,
            "metric_slot": metric_slot, "merge": merge, "temporal": temporal,
            "action_names": actions,
        }

    def manifest_summary(self) -> dict[str, Any]:
        updates = int(self.config["training"]["updates_per_fold"])
        deterministic = int(self.config["training"]["deterministic_population_examples_per_update"])
        return {
            "fold": self.fold_id, "fit_rows": len(self.fit_indices),
            "fit_tracklets": len(self.track_indices), "labeled_tracklets": len(self.labeled_tracks),
            "crossvideo_training_categories": len(self.crossvideo_cats),
            "crossvideo_training_category_ids": self.crossvideo_cats,
            "merge_training_tracklets": len(self.merge_targets),
            "deterministic_examples": updates * deterministic,
            "complete_unique_fit_row_passes": updates * deterministic / len(self.fit_indices),
            "held_categories": sorted(self.held_cats), "calibration_categories": sorted(self.cal_cats),
            "held_videos": sorted(self.held_videos), "held_categories_in_fit": [],
            "calibration_categories_in_fit": [], "held_videos_in_fit": [],
            "input_dim": self.input_dim, "geometry_mean": self.geom_mean.tolist(),
            "geometry_std": self.geom_std.tolist(), "reliability_pos_weight": self.reliability_pos_weight,
        }
