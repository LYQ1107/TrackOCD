"""Common 768-D model inputs containing observed prefixes, never future tails.

Keys/labels/total track length stay in the external scheduler or Train-only
supervision table. This interface makes no frame-online controller claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.trackocd_v2.io import sha256_file

PREFIXES = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class PrefixView:
    visual: np.ndarray
    boxes_normalized_xyxy: np.ndarray
    quality: np.ndarray
    elapsed_frames: np.ndarray

    def weighted_mean(self) -> np.ndarray:
        weights = self.quality
        if weights.sum() <= 1e-12:
            weights = np.ones(len(weights), dtype=np.float32)
        value = (self.visual * weights[:, None]).sum(axis=0) / weights.sum()
        norm = np.linalg.norm(value)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Zero/non-finite causal aggregate")
        return np.asarray(value / norm, dtype=np.float32)


def prefix_view(visual, boxes_normalized_xyxy, quality, frame_ids, observed: int) -> PrefixView:
    """Copy only the visible observations; even validation never reads the tail."""
    if observed < 1:
        raise ValueError("An observation prefix must be nonempty")
    visible = [np.array(a[:observed], dtype=np.float32, copy=True)
               for a in (visual, boxes_normalized_xyxy, quality)]
    frames = np.array(frame_ids[:observed], dtype=np.int64, copy=True)
    if [a.shape for a in visible] != [(observed, 768), (observed, 4), (observed,)] or frames.shape != (observed,):
        raise ValueError("Causal observation arrays are misaligned")
    if any(not np.isfinite(a).all() for a in visible) or np.any(visible[2] < 0):
        raise ValueError("Invalid visible observation")
    if np.any(np.diff(frames) <= 0):
        raise ValueError("Observation frames are not strictly increasing")
    elapsed = frames - frames[0]
    for array in (*visible, elapsed):
        array.flags.writeable = False
    return PrefixView(*visible, elapsed)


class CompactGTFeasibilityCache:
    """Routing metadata is separate; only an observed PrefixView reaches models.

    This reader intentionally cannot promote GT crops to a predicted result.
    It does not load the separate Train supervision table.
    """

    def __init__(self, root: Path):
        import pyarrow.parquet as pq

        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        done = json.loads((root / ".done").read_text())
        if self.manifest["source_role"] != "Train Known GT feasibility, not predicted frontend/main metrics":
            raise ValueError("Not the registered GT-only feasibility cache")
        if done.get("config_sha256") != self.manifest["config"]["sha256"]:
            raise ValueError("Cache completion/config lineage differs")
        for name, record in self.manifest["payloads"].items():
            path = root / name
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError("Cache payload identity mismatch: " + name)
        rows = pq.read_table(root / "index.parquet").to_pylist()
        self._rows = {row["key"]: row for row in rows}
        self._visual = np.load(root / "observations.npy", mmap_mode="r", allow_pickle=False)
        self._geometry = np.load(root / "geometry.npy", mmap_mode="r", allow_pickle=False)
        if len(rows) != len(self._rows) or len(rows) != self.manifest["tracks"]:
            raise ValueError("Compact cache track-index count differs")
        if self._visual.shape != (self.manifest["observations"], 768) or self._visual.dtype != np.float16:
            raise ValueError("Not common 768-D float16 descriptors")

    def keys(self) -> tuple[str, ...]:
        return tuple(self._rows)

    def get_prefix(self, key: str, observed: int) -> PrefixView:
        row = self._rows[key]
        count = min(observed, row["observation_count"])
        start = row["observation_offset"]
        # Do not first materialize the whole track or a full-track aggregate.
        return prefix_view(self._visual[start:start + count], self._geometry[start:start + count],
                           row["quality"][:count], row["frame_ids"][:count], count)
