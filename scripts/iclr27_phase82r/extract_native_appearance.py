#!/usr/bin/env python3
"""Extract corrected DINOv2 appearance for the frozen native Q0 lineage."""
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
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
FRAMES = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/frames")
HUB = Path("/home/user/.cache/torch/hub/facebookresearch_dinov2_main")


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
    ap.add_argument("--shard", type=int, required=True); ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=0); ap.add_argument("--tag", default="native_corrected_r1")
    ap.add_argument("--out", type=Path, required=True); args = ap.parse_args()
    if args.shard < 0 or args.shard >= args.num_shards: raise ValueError("invalid shard")
    meta_path = Path(str(args.out) + ".json")
    if args.out.exists() and meta_path.exists():
        print(json.dumps({"status": "SKIP_EXISTING", "out": str(args.out)})); return
    import torch
    from PIL import Image
    from torchvision import transforms
    rows = [json.loads(line) for line in NATIVE.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [(idx, row) for idx, row in enumerate(rows) if idx % args.num_shards == args.shard]
    if args.max_rows > 0: selected = selected[:args.max_rows]
    tf = transforms.Compose([transforms.Resize((518, 518), interpolation=Image.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = torch.hub.load(str(HUB), "dinov2_vitb14", source="local").eval().to(args.device)
    feats = np.zeros((len(selected), 768), dtype=np.float16); row_indices = np.asarray([idx for idx, _ in selected], dtype=np.int64)
    tensors: list[Any] = []; positions: list[int] = []; failures: list[dict[str, Any]] = []; valid_count = 0
    def flush() -> None:
        nonlocal tensors, positions
        if not tensors: return
        x = torch.cat(tensors, dim=0).to(args.device)
        with torch.no_grad():
            y = torch.nn.functional.normalize(model.forward_features(x)["x_norm_clstoken"], dim=-1).cpu().numpy().astype(np.float32)
        for j, pos in enumerate(positions): feats[pos] = y[j].astype(np.float16)
        tensors, positions = [], []
    for pos, (idx, row) in enumerate(selected):
        if row.get("bbox_xyxy") is None:
            continue
        try:
            with Image.open(FRAMES / str(row["file_path"])) as raw:
                tensors.append(tf(crop_box(raw.convert("RGB"), [float(v) for v in row["bbox_xyxy"]])).unsqueeze(0))
            positions.append(pos); valid_count += 1
            if len(tensors) >= args.batch: flush()
        except Exception as exc:
            failures.append({"row_index": idx, "video_id": row.get("video_id"), "image_id": row.get("image_id"), "error": repr(exc)})
    flush()
    if failures: raise RuntimeError(f"native appearance failures={len(failures)} first={failures[:2]}")
    if not np.isfinite(feats.astype(np.float32)).all(): raise RuntimeError("native features nonfinite")
    args.out.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(args.out) + f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, features=feats, row_indices=row_indices); os.replace(tmp, args.out)
    meta = {"schema_version": "trackocd.phase82r.native_dinov2_corrected_shard.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "shard": args.shard, "num_shards": args.num_shards, "rows": len(selected), "valid_bbox_rows": valid_count, "native_rows": len(rows), "native_sha256": sha256(NATIVE), "frames_root": str(FRAMES), "model_repo": str(HUB), "model": "facebookresearch/dinov2:dinov2_vitb14", "feature": "normalized x_norm_clstoken of causal native Q0 crop", "dtype": "float16", "device": args.device, "batch": args.batch, "no_bbox_rows_zero_filled": True, "future_frames_used": False, "category_or_id_feature": False, "failures": failures, "out_sha256": sha256(args.out)}
    atomic_json(meta_path, meta); print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__": main()
