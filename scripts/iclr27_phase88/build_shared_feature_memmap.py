#!/usr/bin/env python3
"""Materialize one shared read-only feature store for Phase88 workers."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/iclr27_phase88"
SHARED = Path("/data2/usr_for_deadline/trackocd_phase88/shared_features")


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.save(f, value, allow_pickle=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    from src.iclr27_phase19r.data.stream import Phase19RData
    SHARED.mkdir(parents=True, exist_ok=True)
    # Raw and quality are identical fold-wide and are materialized once.
    if not (SHARED / "raw.npy").exists() or not (SHARED / "quality.npy").exists():
        data0 = Phase19RData(0)
        raw = np.asarray(data0.raw, dtype=np.float32)
        quality = np.asarray([data0._quality_row(i) for i in range(len(data0.rows))], dtype=np.float32)
        atomic_npy(SHARED / "raw.npy", raw)
        atomic_npy(SHARED / "quality.npy", quality)
        del data0, raw, quality
        gc.collect()
    fold_meta = {}
    for fold in range(4):
        meta_path = SHARED / f"fold_meta_f{fold}.json"
        required = [meta_path, SHARED / f"geom_f{fold}.npy", SHARED / f"assigned_f{fold}.npy", SHARED / f"row_iou_f{fold}.npy", SHARED / f"known_prototypes_f{fold}.npy", SHARED / f"active_known_mask_f{fold}.npy"]
        if all(p.exists() for p in required):
            fold_meta[str(fold)] = json.loads(meta_path.read_text())
            continue
        data = Phase19RData(fold)
        atomic_npy(SHARED / f"geom_f{fold}.npy", np.asarray(data.geom, dtype=np.float32))
        atomic_npy(SHARED / f"assigned_f{fold}.npy", np.asarray([int(r.get("assigned", 0)) for r in data.rows], dtype=np.int8))
        atomic_npy(SHARED / f"row_iou_f{fold}.npy", np.asarray([float(r.get("row_iou", 0.0)) for r in data.rows], dtype=np.float32))
        atomic_npy(SHARED / f"known_prototypes_f{fold}.npy", np.asarray(data.known_prototypes, dtype=np.float32))
        atomic_npy(SHARED / f"active_known_mask_f{fold}.npy", np.asarray(data.active_known_mask, dtype=bool))
        meta = {
            "fold": fold,
            "track_rows": {str(k): [int(x) for x in v] for k, v in data.track_rows.items()},
            "track_video": {str(k): int(v) for k, v in data.track_video.items()},
            "track_category": {str(k): int(v) for k, v in data.track_category.items()},
            "track_role": {str(k): str(v) for k, v in getattr(data, "track_role", {}).items()},
            "supported_ids": [int(x) for x in data.supported_ids],
            "known_to_index": {str(k): int(v) for k, v in data.known_to_index.items()},
            "known_eval_keys": [[str(k), int(c)] for k, c in __import__("src.iclr27_phase19r.evaluation.internal", fromlist=["fixed_known_keys"]).fixed_known_keys(data)],
        }
        atomic_json(meta_path, meta)
        fold_meta[str(fold)] = meta
        del data
        gc.collect()
    manifest = {
        "schema_version": "trackocd.phase88.shared_features.v1",
        "root": str(SHARED),
        "raw_sha256": sha(SHARED / "raw.npy"),
        "quality_sha256": sha(SHARED / "quality.npy"),
        "folds": {str(k): {"meta_sha256": sha(SHARED / f"fold_meta_f{k}.json")} for k in range(4)},
        "storage": "read_only_numpy_memmap",
        "future_rows_or_tracks": False,
        "ids_or_text_as_model_input": False,
    }
    atomic_json(OUT / "audit/shared_feature_memmap.json", manifest)
    atomic_json(OUT / "completion/build_shared_feature_memmap.done", {"status": "DONE", "manifest": str((OUT / "audit/shared_feature_memmap.json").resolve())})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
