#!/usr/bin/env python3
"""Build compact per-fold track metadata over the shared Phase88 memmaps."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path("/data2/usr_for_deadline/trackocd_phase88/shared_features")
OUT = ROOT / "outputs/iclr27_phase88"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        with open(name, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def main() -> None:
    manifest = {}
    for fold in range(4):
        meta_path = SHARED / f"fold_meta_f{fold}.json"
        meta = json.loads(meta_path.read_text())
        keys = sorted(meta["track_rows"])
        offsets = np.zeros(len(keys) + 1, dtype=np.int64)
        flat: list[int] = []
        videos = np.zeros(len(keys), dtype=np.int32)
        categories = np.zeros(len(keys), dtype=np.int32)
        roles: dict[str, int] = {}
        role_codes = np.zeros(len(keys), dtype=np.int8)
        role_names = sorted(set(str(v) for v in meta.get("track_role", {}).values()))
        role_to_code = {name: i for i, name in enumerate(role_names)}
        for i, key in enumerate(keys):
            rows = [int(x) for x in meta["track_rows"][key]]
            flat.extend(rows)
            offsets[i + 1] = len(flat)
            videos[i] = int(meta["track_video"][key])
            categories[i] = int(meta["track_category"][key])
            role_codes[i] = role_to_code.get(str(meta.get("track_role", {}).get(key, "")), -1)
        prefix = SHARED / f"track_index_f{fold}"
        atomic_json(prefix.with_name(prefix.name + ".json"), {
            "fold": fold, "keys": keys, "role_to_code": role_to_code,
            "storage": "compact_offsets_indices_arrays",
        })
        atomic_npy(prefix.with_name(prefix.name + "_offsets.npy"), offsets)
        atomic_npy(prefix.with_name(prefix.name + "_indices.npy"), np.asarray(flat, dtype=np.int64))
        atomic_npy(prefix.with_name(prefix.name + "_video.npy"), videos)
        atomic_npy(prefix.with_name(prefix.name + "_category.npy"), categories)
        atomic_npy(prefix.with_name(prefix.name + "_role.npy"), role_codes)
        manifest[str(fold)] = {
            "track_count": len(keys),
            "membership_count": len(flat),
            "keys": str(prefix.with_name(prefix.name + ".json")),
        }
    atomic_json(OUT / "audit/compact_track_index.json", {
        "schema_version": "trackocd.phase88.compact_track_index.v1",
        "root": str(SHARED), "folds": manifest,
        "future_rows_or_tracks": False, "ids_or_text_as_model_input": False,
    })
    atomic_json(OUT / "completion/build_compact_track_index.done", {"status": "DONE"})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
