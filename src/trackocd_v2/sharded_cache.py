"""Reader for the TrackOCD v2 sharded feature cache.

The reader keeps the parquet lineage index in memory and memory-maps only the
currently requested shard.  It exposes the same small feature dictionary used
by the legacy per-track JSON baselines, so methods do not need to know whether
the formal cache is stored as JSON or sharded NumPy arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np


PREFIXES = (1, 2, 4, 8, 16)


class ShardedFeatureStore:
    """Open a completed ``cache_manifest.json`` without copying feature data."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.manifest: Dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "COMPLETE":
            raise RuntimeError("sharded feature cache is not complete: %s" % self.manifest.get("status"))
        if self.manifest.get("format") != "sharded_numpy_with_parquet_index":
            raise ValueError("unsupported sharded feature format")
        if tuple(int(value) for value in self.manifest.get("prefixes", [])) != PREFIXES:
            raise ValueError("sharded feature prefix contract is not 1/2/4/8/16")
        self.cache_root = Path(str(self.manifest["cache_root"])).resolve()
        self.shard_tracks = int(self.manifest["shard_tracks"])
        if self.shard_tracks < 1:
            raise ValueError("invalid shard_tracks")
        self._entries: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
        self._shards: Dict[int, Path] = {}
        self._current_index: Optional[int] = None
        self._current_observations: Optional[np.ndarray] = None
        self._current_prefixes: Optional[np.ndarray] = None
        self._current_full: Optional[np.ndarray] = None
        for item in self.manifest.get("completed_shards", []):
            track_start = int(item["track_start"])
            shard_index = track_start // self.shard_tracks
            shard_dir = self.cache_root / "shards" / ("shard-%06d" % shard_index)
            self._shards[shard_index] = shard_dir
            self._load_index(shard_index, shard_dir, int(item["track_count"]))
        expected = int(self.manifest.get("total_tracks", -1))
        if expected < 0 or len(self._entries) != expected:
            raise ValueError("sharded index coverage mismatch: %d != %d" % (len(self._entries), expected))

    def _load_index(self, shard_index: int, shard_dir: Path, expected_rows: int) -> None:
        index_path = shard_dir / "index.parquet"
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read the sharded feature index") from exc
        rows = parquet.read_table(str(index_path)).to_pylist()
        if len(rows) != expected_rows:
            raise ValueError("index row count mismatch in %s" % index_path)
        for row_index, row in enumerate(rows):
            key = str(row["sample_key"])
            if key in self._entries:
                raise ValueError("duplicate sample_key in sharded cache: %s" % key)
            self._entries[key] = (shard_index, row_index, row)

    def _load_shard(self, shard_index: int) -> None:
        if self._current_index == shard_index:
            return
        shard_dir = self._shards[shard_index]
        self._current_observations = np.load(shard_dir / "observations.npy", mmap_mode="r", allow_pickle=False)
        self._current_prefixes = np.load(shard_dir / "prefix_features.npy", mmap_mode="r", allow_pickle=False)
        self._current_full = np.load(shard_dir / "full_features.npy", mmap_mode="r", allow_pickle=False)
        if self._current_observations.ndim != 2 or self._current_observations.shape[1] != 768:
            raise ValueError("invalid observation array shape in %s" % shard_dir)
        if self._current_prefixes.ndim != 3 or self._current_prefixes.shape[1:] != (len(PREFIXES), 768):
            raise ValueError("invalid prefix array shape in %s" % shard_dir)
        if self._current_full.ndim != 2 or self._current_full.shape[1] != 768:
            raise ValueError("invalid full array shape in %s" % shard_dir)
        if self._current_prefixes.shape[0] != self._current_full.shape[0]:
            raise ValueError("prefix/full track count mismatch in %s" % shard_dir)
        self._current_index = shard_index

    def get(self, sample_key: str) -> Dict[str, Any]:
        key = str(sample_key)
        try:
            shard_index, row_index, row = self._entries[key]
        except KeyError as exc:
            raise KeyError("sample key is absent from sharded cache: %s" % key) from exc
        self._load_shard(shard_index)
        assert self._current_observations is not None
        assert self._current_prefixes is not None
        assert self._current_full is not None
        offset = int(row["observation_offset"])
        count = int(row["observation_count"])
        if offset < 0 or count < 1:
            raise ValueError("observation offset/length is invalid for %s" % key)
        observations = self._current_observations[offset:offset + count]
        if observations.shape != (count, 768):
            raise ValueError("observation offset/length is invalid for %s" % key)
        prefix_matrix = self._current_prefixes[row_index]
        full = self._current_full[row_index]
        if prefix_matrix.shape != (len(PREFIXES), 768) or full.shape != (768,):
            raise ValueError("feature row shape is invalid for %s" % key)
        return {
            "sample_key": key,
            "source_split": str(row.get("source_split", "")),
            "video_id": int(row["video_id"]),
            "physical_track_id": str(row["physical_track_id"]),
            "frame_ids": [int(value) for value in row["frame_ids"]],
            "image_paths": [str(value) for value in row["image_paths"]],
            "boxes_xyxy": row["boxes_xyxy"],
            "quality": [float(value) for value in row["quality"]],
            "frame_embeddings": np.asarray(observations, dtype=np.float32),
            "prefix_features": {
                str(prefix): np.asarray(prefix_matrix[index], dtype=np.float32)
                for index, prefix in enumerate(PREFIXES)
            },
            "full_feature": np.asarray(full, dtype=np.float32),
            "prefix_observations_used": {
                str(prefix): int(row.get("prefix_observations_used", [min(prefix, count) for prefix in PREFIXES])[index])
                for index, prefix in enumerate(PREFIXES)
            },
        }

    def __contains__(self, sample_key: str) -> bool:
        return str(sample_key) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> Iterator[str]:
        for key, _value in sorted(self._entries.items(), key=lambda item: int(item[1][2].get("global_track_index", 0))):
            yield key
