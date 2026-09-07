"""Lightweight read-only Phase88 fold view backed by shared NumPy memmaps."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/data2/usr_for_deadline/trackocd_phase88/shared_features")


class Phase88FoldData:
    def __init__(self, fold: int, root: str | Path = ROOT):
        self.fold = int(fold)
        self.root = Path(root)
        meta = json.loads((self.root / f"fold_meta_f{self.fold}.json").read_text())
        self.rows = None
        self.track_rows = {str(k): [int(x) for x in v] for k, v in meta["track_rows"].items()}
        self.track_video = {str(k): int(v) for k, v in meta["track_video"].items()}
        self.track_category = {str(k): int(v) for k, v in meta["track_category"].items()}
        self.track_role = {str(k): str(v) for k, v in meta.get("track_role", {}).items()}
        self.supported_ids = [int(x) for x in meta["supported_ids"]]
        self.known_to_index = {int(k): int(v) for k, v in meta["known_to_index"].items()}
        self.known_eval_keys = [(str(k), int(c)) for k, c in meta["known_eval_keys"]]
        self.raw = np.load(self.root / "raw.npy", mmap_mode="r")
        self.quality = np.load(self.root / "quality.npy", mmap_mode="r")
        self.geom = np.load(self.root / f"geom_f{self.fold}.npy", mmap_mode="r")
        self._assigned = np.load(self.root / f"assigned_f{self.fold}.npy", mmap_mode="r")
        self._row_iou = np.load(self.root / f"row_iou_f{self.fold}.npy", mmap_mode="r")
        self.known_prototypes = np.load(self.root / f"known_prototypes_f{self.fold}.npy", mmap_mode="r")
        self.active_known_mask = np.load(self.root / f"active_known_mask_f{self.fold}.npy", mmap_mode="r")

    def _quality_row(self, row_index: int) -> float:
        return float(self.quality[int(row_index)])

    def prefix(self, key: str):
        indices = self.track_rows[key]
        idx = np.asarray(indices[:16], dtype=np.int64)
        return np.asarray(self.raw[idx]), np.asarray(self.geom[idx]), np.asarray(self.quality[idx]), np.ones(len(idx), dtype=bool)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "rows": int(self.raw.shape[0]),
            "tracklets": len(self.track_rows),
            "known_count": len(self.supported_ids),
            "active_known_count": int(np.asarray(self.active_known_mask).sum()),
            "storage": "read_only_numpy_memmap",
            "future_rows_or_tracks": False,
            "ids_or_text_as_model_input": False,
        }
