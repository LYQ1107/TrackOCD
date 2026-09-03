#!/usr/bin/env python3
"""Merge deterministic Phase82P DINOv2 row-index shards without copying Q0."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q0 = ROOT / "outputs/iclr27_phase4t/train_stream/teta/tao_track.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = json.loads(Q0.read_text(encoding="utf-8"))
    all_idx: list[np.ndarray] = []
    all_feat: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    for path in args.shards:
        z = np.load(path, allow_pickle=False)
        idx = np.asarray(z["row_indices"], dtype=np.int64)
        feat = np.asarray(z["features"], dtype=np.float16)
        if len(idx) != len(feat) or len(np.unique(idx)) != len(idx):
            raise RuntimeError(f"invalid shard {path}")
        all_idx.append(idx); all_feat.append(feat)
        meta_path = Path(str(path) + ".json")
        metas.append(json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {"path": str(path)})
    indices = np.concatenate(all_idx); features = np.concatenate(all_feat)
    order = np.argsort(indices)
    indices, features = indices[order], features[order]
    expected = np.arange(len(rows), dtype=np.int64)
    if not np.array_equal(indices, expected):
        raise RuntimeError(f"shards do not exactly cover Q0 rows: {len(indices)} vs {len(rows)}")
    if not np.isfinite(features.astype(np.float32)).all():
        raise RuntimeError("merged features contain non-finite values")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, features=features)
    os.replace(tmp, args.out)
    meta = {
        "schema_version": "trackocd.phase82p.q0_dinov2_appearance.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "q0_path": str(Q0), "q0_sha256": sha256(Q0), "rows": len(rows),
        "shape": list(features.shape), "dtype": str(features.dtype),
        "shards": metas, "out_sha256": sha256(args.out),
        "feature": "normalized DINOv2 CLS of current causal Q0 proposal crop",
        "future_frames_used": False, "category_or_id_feature": False,
    }
    atomic_json(Path(str(args.out) + ".json"), meta)
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
