#!/usr/bin/env python3
"""Merge native appearance shards by immutable native row index."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")


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
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", nargs="+", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    n = sum(1 for line in NATIVE.open() if line.strip())
    indices = []; features = []; metas = []
    for path in args.shards:
        z = np.load(path, allow_pickle=False); idx = np.asarray(z["row_indices"], dtype=np.int64); feat = np.asarray(z["features"], dtype=np.float16)
        if len(idx) != len(feat) or len(np.unique(idx)) != len(idx): raise RuntimeError(f"invalid shard {path}")
        indices.append(idx); features.append(feat)
        side = Path(str(path) + ".json"); metas.append(json.loads(side.read_text()) if side.is_file() else {"path": str(path)})
    idx = np.concatenate(indices); feat = np.concatenate(features); order = np.argsort(idx); idx, feat = idx[order], feat[order]
    if not np.array_equal(idx, np.arange(n, dtype=np.int64)): raise RuntimeError(f"coverage mismatch {len(idx)} != {n}")
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(args.out) + f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, features=feat); os.replace(tmp, args.out)
    atomic_json(Path(str(args.out) + ".json"), {"schema_version": "trackocd.phase82r.native_dinov2_corrected.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "native_path": str(NATIVE), "native_sha256": sha256(NATIVE), "rows": n, "shape": list(feat.shape), "dtype": str(feat.dtype), "shards": metas, "out_sha256": sha256(args.out), "no_bbox_rows_zero_filled": True, "future_frames_used": False, "category_or_id_feature": False})
    print(json.dumps({"rows": n, "shape": list(feat.shape), "out": str(args.out), "out_sha256": sha256(args.out)}, indent=2))


if __name__ == "__main__": main()
