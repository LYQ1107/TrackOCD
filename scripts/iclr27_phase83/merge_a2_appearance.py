#!/usr/bin/env python3
"""Atomically merge complete A2 DINOv2 feature shards by lineage index."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    n = sum(1 for line in args.native.open(encoding="utf-8") if line.strip())
    if len(args.shards) != 4:
        raise ValueError("A2 merge requires four shards")
    indices, features, sides = [], [], []
    for path in args.shards:
        z = np.load(path, allow_pickle=False)
        idx = np.asarray(z["row_indices"], dtype=np.int64)
        feat = np.asarray(z["features"], dtype=np.float16)
        if len(idx) != len(feat) or len(np.unique(idx)) != len(idx) or feat.shape[1:] != (768,):
            raise RuntimeError(f"invalid shard {path}")
        indices.append(idx); features.append(feat)
        side = Path(str(path) + ".json")
        if not side.is_file():
            raise FileNotFoundError(side)
        sides.append(json.loads(side.read_text(encoding="utf-8")))
    idx = np.concatenate(indices); feat = np.concatenate(features)
    order = np.argsort(idx); idx, feat = idx[order], feat[order]
    if not np.array_equal(idx, np.arange(n, dtype=np.int64)):
        raise RuntimeError(f"coverage mismatch: {len(idx)} rows for {n} native rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, features=feat)
    os.replace(tmp, args.out)
    atomic_json(Path(str(args.out) + ".json"), {
        "schema_version": "trackocd.phase83.a2_dinov2_corrected.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "native": str(args.native.resolve()), "native_sha256": sha256(args.native),
        "rows": n, "shape": list(feat.shape), "dtype": str(feat.dtype),
        "shards": sides, "out_sha256": sha256(args.out), "future_frames_used": False,
        "category_or_id_feature": False, "row_index_coverage": "exact_0_to_n_minus_1",
    })
    print(json.dumps({"status": "COMPLETE", "rows": n, "shape": list(feat.shape), "out": str(args.out), "out_sha256": sha256(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
