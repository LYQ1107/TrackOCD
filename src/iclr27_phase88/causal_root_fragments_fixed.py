"""Phase87 correction of the Phase86 canonical-root observed-step contract.

All causal cutoffs are ordinal observed-frame steps, never native image IDs.
Phase86 sources remain read-only; this module is the corrected copy.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from scripts.iclr27_phase85.build_physical_r_adapter import join_rows
from src.iclr27_phase23.protocol import order_key, track_key


@dataclass(frozen=True)
class UnionEvent:
    video_id: int
    observed_step: int
    child_id: int
    parent_id: int
    frame_id: int
    image_id: int


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, value: int) -> int:
        if value not in self.parent:
            self.parent[value] = value
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


class CausalUnionTimeline:
    def __init__(self, lineage: list[dict[str, Any]], events: list[UnionEvent]) -> None:
        self.lineage = lineage
        self.rows_by_video: dict[int, list[int]] = defaultdict(list)
        self.events_by_video: dict[int, list[UnionEvent]] = defaultdict(list)
        self.ids_by_video: dict[int, set[int]] = defaultdict(set)
        for index, row in enumerate(lineage):
            video = int(row.get("video_id", -1))
            original = int(row.get("original_physical_track_id", -1))
            self.rows_by_video[video].append(index)
            self.ids_by_video[video].add(original)
        for event in events:
            self.events_by_video[event.video_id].append(event)
            self.ids_by_video[event.video_id].update((event.child_id, event.parent_id))
        for video in self.rows_by_video:
            self.rows_by_video[video].sort(key=lambda i: (int(lineage[i].get("frame_id", 0)), int(lineage[i].get("image_id", 0)), i))
        for video in self.events_by_video:
            self.events_by_video[video].sort(key=lambda e: (e.observed_step, e.child_id, e.parent_id))

    def members_at(self, video_id: int, cutoff_step: int, anchor_original_id: int) -> set[int]:
        dsu = DisjointSet()
        for value in self.ids_by_video.get(video_id, set()):
            dsu.find(value)
        for event in self.events_by_video.get(video_id, []):
            if event.observed_step > cutoff_step:
                break
            dsu.union(event.child_id, event.parent_id)
        root = dsu.find(anchor_original_id)
        return {value for value in self.ids_by_video.get(video_id, set()) if dsu.find(value) == root}


class FragmentSetBuilder:
    def __init__(self, table, public_rows, lineage, timeline, mapped, mapped_iou) -> None:
        self.table = table
        self.public_rows = public_rows
        self.lineage = lineage
        self.timeline = timeline
        self.mapped = mapped
        self.mapped_iou = mapped_iou
        self.by_track = defaultdict(list)
        self.native_to_public = defaultdict(list)
        for index, row in enumerate(public_rows):
            self.by_track[track_key(row)].append(index)
        for key in self.by_track:
            self.by_track[key].sort(key=lambda i: order_key(public_rows[i]))
        for index, native_index in enumerate(mapped):
            if native_index >= 0 and mapped_iou[index] >= 0.5:
                self.native_to_public[int(native_index)].append(index)
        self.original_rows = defaultdict(list)
        for native_index, public_indices in self.native_to_public.items():
            original = int(lineage[native_index].get("original_physical_track_id", -1))
            self.original_rows[original].extend(public_indices)

    def build(self, track_key_value: str, prefix: int) -> dict[str, Any]:
        sequence = self.by_track.get(track_key_value, [])
        used = sequence[: min(int(prefix), len(sequence))]
        reliable = [i for i in used if self.mapped[i] >= 0 and self.mapped_iou[i] >= 0.5]
        raw = self.table.raw_vector(track_key_value, prefix)
        video_id = int(self.table.metadata[track_key_value]["video"])
        if not reliable:
            return {
                "track_key": track_key_value,
                "prefix": int(prefix),
                "video_id": video_id,
                "anchor_original_id": None,
                "cutoff_step": None,
                "fragment_ids": [],
                "fragment_vectors": np.zeros((0, 768), np.float32),
                "raw_anchor_vector": raw,
                "fallback": True,
            }
        anchor_public = reliable[-1]
        native_index = int(self.mapped[anchor_public])
        native_row = self.lineage[native_index]
        video_id = int(native_row.get("video_id", -1))
        cutoff_step = int(native_row.get("_observed_step", 0))
        anchor_original = int(native_row.get("original_physical_track_id", -1))
        members = sorted(self.timeline.members_at(video_id, cutoff_step, anchor_original))
        vectors: list[np.ndarray] = []
        present: list[int] = []
        for original in members:
            public_indices = [
                i
                for i in self.original_rows.get(original, [])
                if int(self.public_rows[i].get("video_id", -1)) == video_id
                and int(self.public_rows[i].get("_observed_step", 0)) <= cutoff_step
            ]
            if not public_indices:
                continue
            array = self.table.features[np.asarray(public_indices, dtype=np.int64)]
            array = array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)
            vector = array.mean(axis=0)
            vector = vector / max(float(np.linalg.norm(vector)), 1e-8)
            vectors.append(vector.astype(np.float32))
            present.append(original)
        return {
            "track_key": track_key_value,
            "prefix": int(prefix),
            "video_id": video_id,
            "anchor_original_id": anchor_original,
            "cutoff_step": cutoff_step,
            "fragment_ids": present,
            "fragment_vectors": np.asarray(vectors, np.float32).reshape((-1, 768)) if vectors else np.zeros((0, 768), np.float32),
            "raw_anchor_vector": raw,
            "fallback": not bool(vectors),
        }


def load_timeline(lineage_path, union_path):
    import json

    lineage = [json.loads(line) for line in lineage_path.open() if line.strip()]
    frame_keys: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in lineage:
        frame_keys[int(row.get("video_id", -1))].append((int(row.get("frame_id", 0)), int(row.get("image_id", 0))))
    step_by_frame = {
        video: {key: step for step, key in enumerate(sorted(set(keys)))}
        for video, keys in frame_keys.items()
    }
    for row in lineage:
        video = int(row.get("video_id", -1))
        key = (int(row.get("frame_id", 0)), int(row.get("image_id", 0)))
        row["_observed_step"] = int(step_by_frame[video][key])
    events: list[UnionEvent] = []
    missing = 0
    for line in union_path.open():
        if not line.strip():
            continue
        obj = json.loads(line)
        video = int(obj["video_id"])
        key = (int(obj.get("frame_id", 0)), int(obj.get("image_id", 0)))
        if key not in step_by_frame.get(video, {}):
            missing += 1
            continue
        events.append(UnionEvent(video, step_by_frame[video][key], int(obj["child_original_physical_track_id"]), int(obj["parent_canonical_physical_track_id"]), key[0], key[1]))
    timeline = CausalUnionTimeline(lineage, events)
    audit = {"lineage_rows": len(lineage), "union_events": len(events), "union_events_missing_observed_frame": missing, "step_rule": "sorted(frame_id,image_id) ordinal per video"}
    return lineage, timeline, audit


def make_builder(table, public_rows, lineage_path, union_path):
    lineage, timeline, audit = load_timeline(lineage_path, union_path)
    mapped, mapped_iou, _, join_audit = join_rows(public_rows, lineage)
    for index, row in enumerate(public_rows):
        native_index = int(mapped[index])
        row["_observed_step"] = int(lineage[native_index].get("_observed_step", 0)) if native_index >= 0 else -1
    audit["join"] = join_audit
    return FragmentSetBuilder(table, public_rows, lineage, timeline, mapped, mapped_iou), audit
