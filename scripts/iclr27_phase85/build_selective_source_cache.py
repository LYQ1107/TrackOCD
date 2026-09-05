#!/usr/bin/env python3
"""Build a source-track cache from the Phase85 selective physical lineage.

This is the registered P1+S0 diagnostic: source representations follow the
selective lineage's causal canonical root, while all features remain the
frozen DINOv2 rows.  Labels are not used to construct model inputs.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key
from scripts.iclr27_phase85.build_physical_r_adapter import join_rows, parse_box, as_int

NATIVE = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs/physical/selective_formal_r1/full_temporal_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = Path("/data2/usr_for_deadline/trackocd_phase85/project_outputs")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def main() -> None:
    table = load_frozen_tracks()
    public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8")))
    native = [json.loads(line) for line in NATIVE.open(encoding="utf-8") if line.strip()]
    features = np.asarray(np.load(FEATURES, allow_pickle=False)["features"], np.float32)
    if len(features) != len(native):
        raise RuntimeError(f"feature/native length mismatch {len(features)} != {len(native)}")
    mapped, mapped_iou, _, join = join_rows(public, native)
    roots = {
        i: (as_int(row.get("video_id")), as_int(row.get("phase85_canonical_physical_track_id", row.get("physical_track_id"))))
        for i, row in enumerate(native)
    }
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(public):
        tracks[track_key(row)].append(i)
    for key in tracks:
        tracks[key].sort(key=lambda i: order_key(public[i]))
    # Public-row membership indexed by selective canonical root.  We only use
    # rows with an explicit IoU-validated native join.
    root_rows: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for i, nidx in enumerate(mapped):
        if nidx >= 0 and float(mapped_iou[i]) >= 0.5:
            root_rows[roots[int(nidx)]].append((i, int(nidx)))
    for root in root_rows:
        root_rows[root].sort(key=lambda pair: order_key(public[pair[0]]))

    keys = sorted(table.metadata)
    key_index = {key: i for i, key in enumerate(keys)}
    vectors = np.zeros((len(PREFIXES), len(keys), 768), np.float32)
    raw = np.stack([[table.raw_vector(key, p) for key in keys] for p in PREFIXES]).astype(np.float32)
    prototypes = np.zeros((3, len(keys), 768), np.float32)
    coverage = {str(p): 0 for p in PREFIXES}
    fallback = {str(p): 0 for p in PREFIXES}
    for pi, prefix in enumerate(PREFIXES):
        for key in keys:
            seq = tracks.get(key, [])
            use = seq[:min(prefix, len(seq))]
            good = [i for i in use if mapped[i] >= 0 and float(mapped_iou[i]) >= 0.5]
            if not good:
                vectors[pi, key_index[key]] = raw[pi, key_index[key]]
                fallback[str(prefix)] += 1
                continue
            anchor = good[-1]
            anchor_root = roots[int(mapped[anchor])]
            cutoff = order_key(public[anchor])
            members = [nidx for pub_i, nidx in root_rows.get(anchor_root, []) if order_key(public[pub_i]) <= cutoff]
            if not members:
                vectors[pi, key_index[key]] = raw[pi, key_index[key]]
                fallback[str(prefix)] += 1
                continue
            arr = features[np.asarray(members, np.int64)]
            vectors[pi, key_index[key]] = norm(norm(arr).mean(axis=0))
            coverage[str(prefix)] += 1
            if prefix == 16:
                for ci, chunk in enumerate(np.array_split(arr, 3)):
                    if len(chunk):
                        prototypes[ci, key_index[key]] = norm(chunk.mean(axis=0))
    data = OUT / "manifests/source_track_selective_vectors.npz"
    data.parent.mkdir(parents=True, exist_ok=True)
    tmp = data.with_name(f".{data.name}.{os.getpid()}.tmp.npz")
    np.savez(tmp, keys=np.asarray(keys), vectors=vectors, prototypes=prototypes, raw_vectors=raw)
    os.replace(tmp, data)
    manifest = {
        "schema_version": "trackocd.phase85.source_selective_cache.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "lineage": str(NATIVE.resolve()), "lineage_sha256": sha(NATIVE),
        "native_features": str(FEATURES.resolve()), "native_features_sha256": sha(FEATURES),
        "public_csv": str(PUBLIC.resolve()), "public_csv_sha256": sha(PUBLIC),
        "data": str(data.resolve()), "data_sha256": sha(data),
        "track_count": len(keys), "prefix_coverage": coverage, "prefix_fallback": fallback,
        "join": join, "membership": "single selective canonical root at last causal mapped row; same-video and <= cutoff",
        "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False,
        "ids_as_model_input": False, "labels_posthoc_only": True,
    }
    atomic_json(OUT / "manifests/source_track_selective_vectors.json", manifest)
    atomic_json(OUT / "audit/source_selective_cache.json", manifest)
    print(json.dumps({"status": "DONE", "tracks": len(keys), "coverage": coverage, "fallback": fallback, "join": {k: join[k] for k in ("public_rows", "mapped_rows", "exact_rows", "fallback_rows", "unmatched_rows")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
