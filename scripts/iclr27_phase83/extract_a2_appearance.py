#!/usr/bin/env python3
"""Extract corrected causal DINOv2 features for the complete A2 Q0 trace."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def crop_box(image: Any, box: list[float], context: float = 0.10) -> Any:
    w, h = image.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    xa = max(0.0, cx - bw * (1.0 + 2.0 * context) * 0.5)
    ya = max(0.0, cy - bh * (1.0 + 2.0 * context) * 0.5)
    xb = min(float(w), cx + bw * (1.0 + 2.0 * context) * 0.5)
    yb = min(float(h), cy + bh * (1.0 + 2.0 * context) * 0.5)
    if xb - xa < 2 or yb - ya < 2:
        xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    return image.crop((int(xa), int(ya), int(xb), int(yb)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-repo", type=Path, default=Path("/home/user/.cache/torch/hub/facebookresearch_dinov2_main"))
    args = ap.parse_args()
    if not args.native.is_file() or not args.frames.is_dir():
        raise FileNotFoundError("native lineage or frames root missing")
    if args.shard < 0 or args.shard >= args.num_shards:
        raise ValueError("invalid shard")
    side = Path(str(args.out) + ".json")
    if args.out.exists() and side.exists():
        print(json.dumps({"status": "SKIP_EXISTING", "out": str(args.out)}))
        return
    import torch
    from PIL import Image
    from torchvision import transforms

    rows = [json.loads(line) for line in args.native.open(encoding="utf-8") if line.strip()]
    selected = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard]
    transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = torch.hub.load(str(args.model_repo), "dinov2_vitb14", source="local").eval().to(args.device)
    feats = np.zeros((len(selected), 768), dtype=np.float16)
    row_indices = np.asarray([i for i, _ in selected], dtype=np.int64)
    tensors: list[Any] = []
    positions: list[int] = []
    valid_bbox = 0

    def flush() -> None:
        nonlocal tensors, positions
        if not tensors:
            return
        x = torch.cat(tensors, dim=0).to(args.device)
        with torch.no_grad():
            z = torch.nn.functional.normalize(model.forward_features(x)["x_norm_clstoken"], dim=-1).cpu().numpy().astype(np.float32)
        for j, pos in enumerate(positions):
            feats[pos] = z[j].astype(np.float16)
        tensors, positions = [], []

    for pos, (idx, row) in enumerate(selected):
        box = row.get("bbox_xyxy")
        if not box:
            continue
        try:
            with Image.open(args.frames / str(row["file_path"])) as raw:
                tensors.append(transform(crop_box(raw.convert("RGB"), [float(v) for v in box])).unsqueeze(0))
            positions.append(pos)
            valid_bbox += 1
            if len(tensors) >= args.batch:
                flush()
        except Exception as exc:
            raise RuntimeError(f"row {idx} appearance extraction failed: {exc!r}") from exc
    flush()
    if valid_bbox and not np.isfinite(feats.astype(np.float32)).all():
        raise RuntimeError("non-finite appearance features")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, features=feats, row_indices=row_indices)
    os.replace(tmp, args.out)
    atomic_json(side, {
        "schema_version": "trackocd.phase83.a2_dinov2_corrected_shard.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "native": str(args.native.resolve()), "native_sha256": sha256(args.native),
        "frames_root": str(args.frames.resolve()), "shard": args.shard, "num_shards": args.num_shards,
        "rows": len(selected), "native_rows": len(rows), "valid_bbox_rows": valid_bbox,
        "model_repo": str(args.model_repo.resolve()), "model": "facebookresearch/dinov2:dinov2_vitb14",
        "feature": "normalized x_norm_clstoken from causal native Q0 bbox crop",
        "dtype": "float16", "device": args.device, "batch": args.batch,
        "termination_rows_zero_filled": len(selected) - valid_bbox, "future_frames_used": False,
        "category_or_id_feature": False, "out_sha256": sha256(args.out),
    })
    print(json.dumps({"status": "COMPLETE", "shard": args.shard, "rows": len(selected), "valid_bbox_rows": valid_bbox, "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
