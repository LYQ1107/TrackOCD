#!/usr/bin/env python3
"""Extract deterministic DINOv3 CLS + dense patch tokens for one row shard."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.iclr27_phase80a.dense_source import DinoV3Dense, atomic_json, crop_box, load_rows, make_transform
from src.iclr27_phase80a.dense_source import row_key

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase80a"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp + ".npz", path)
    finally:
        for candidate in (tmp, tmp + ".npz"):
            if os.path.exists(candidate):
                os.unlink(candidate)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--cache-root", default="/data2/usr_for_deadline/trackocd_phase80a/dense_cache")
    args = ap.parse_args()
    if args.shard < 0 or args.shard >= args.num_shards:
        raise SystemExit("invalid shard")
    marker = OUT / "completion" / f"dense_shard_{args.shard}.launched"
    done = OUT / "completion" / f"dense_shard_{args.shard}.done"
    if done.exists():
        print(json.dumps({"status": "already_done", "shard": args.shard}))
        return
    if marker.exists():
        raise SystemExit(f"refusing duplicate launched unit: {marker}")
    started = time.time()
    atomic_text(marker, json.dumps({"pid": os.getpid(), "shard": args.shard, "started_unix": started}))
    rows = load_rows()
    start, end = (len(rows) * args.shard) // args.num_shards, (len(rows) * (args.shard + 1)) // args.num_shards
    local_rows = rows[start:end]
    cache_root = Path(args.cache_root)
    out_path = cache_root / f"dense_shard_{args.shard:02d}.npz"
    if out_path.exists():
        atomic_json(done, {"status": "cache_exists", "shard": args.shard, "path": str(out_path.resolve()), "rows": len(local_rows)})
        return
    try:
        torch.set_num_threads(2)
        encoder = DinoV3Dense(device=args.device)
        transform = make_transform()
        cls = np.zeros((len(local_rows), encoder.feature_dim), dtype=np.float16)
        patch = np.zeros((len(local_rows), 32, encoder.feature_dim), dtype=np.float16)
        valid = np.zeros(len(local_rows), dtype=np.uint8)
        keys: list[str] = []
        image_ids = np.zeros(len(local_rows), dtype=np.int64)
        batch_tensors: list[torch.Tensor] = []
        batch_indices: list[int] = []

        def flush() -> None:
            nonlocal batch_tensors, batch_indices
            if not batch_tensors:
                return
            x = torch.stack(batch_tensors, dim=0)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=str(args.device).startswith("cuda")):
                c, p = encoder.encode(x)
            for j, idx in enumerate(batch_indices):
                cls[idx] = c[j].astype(np.float16)
                patch[idx] = p[j].astype(np.float16)
                valid[idx] = 1
            batch_tensors, batch_indices = [], []

        for local_idx, row in enumerate(local_rows):
            keys.append(row_key(row))
            image_ids[local_idx] = int(row.get("image_id", -1))
            try:
                image_path = ROOT / "data/raw/tao/frames" / row["image_path"]
                with Image.open(image_path) as image:
                    crop = crop_box(image.convert("RGB"), json.loads(row["bbox_xyxy"]))
                if min(crop.size) < 4:
                    continue
                batch_tensors.append(transform(crop))
                batch_indices.append(local_idx)
                if len(batch_tensors) >= args.batch:
                    flush()
            except Exception as exc:
                print(json.dumps({"row": local_idx + start, "error": repr(exc)}), flush=True)
        flush()
        if int(valid.sum()) != len(local_rows):
            raise RuntimeError(f"invalid crop rows: {len(local_rows) - int(valid.sum())}")
        atomic_npz(out_path, row_keys=np.asarray(keys), image_ids=image_ids, cls=cls, patch=patch, valid=valid)
        atomic_json(done, {"status": "done", "shard": args.shard, "rows": len(local_rows), "valid": int(valid.sum()), "path": str(out_path.resolve()), "elapsed_sec": time.time() - started})
        print(json.dumps({"status": "done", "shard": args.shard, "rows": len(local_rows), "seconds": round(time.time() - started, 2)}), flush=True)
    except Exception as exc:
        atomic_json(OUT / "completion" / f"dense_shard_{args.shard}.failed.json", {"status": "failed", "shard": args.shard, "error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
