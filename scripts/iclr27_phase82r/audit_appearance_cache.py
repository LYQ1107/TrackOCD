#!/usr/bin/env python3
"""Small, read-only audit of the Phase82P DINOv2 appearance cache.

This intentionally evaluates a handful of distinct RGB crops directly with the
same local DINOv2 checkpoint and compares them with the cached row vectors.  It
does not train, access held labels, or alter any Phase82P artifact.
"""
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
ANN = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/TAO-Amodal/annotations/train.json")
FRAMES = ANN.parent.parent / "frames"
CACHE = ROOT / "outputs/iclr27_phase82p/features/q0_dinov2.npz"
HUB = Path("/home/user/.cache/torch/hub/facebookresearch_dinov2_main")
OUT = ROOT / "outputs/iclr27_phase82r/audit/appearance_cache_audit.json"


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / den) if den > 1e-8 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--indices", default="0,1,20,100,1000,10000")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision import transforms

    rows = json.loads(Q0.read_text(encoding="utf-8"))
    ann = json.loads(ANN.read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in ann["images"]}
    cached = np.asarray(np.load(CACHE, mmap_mode="r")["features"], dtype=np.float32)
    indices = [int(x) for x in args.indices.split(",") if x.strip()]
    if any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError("sample index outside Q0 rows")
    tf = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    model = torch.hub.load(str(HUB), "dinov2_vitb14", source="local").eval().to(args.device)
    tensors = []
    pixel_stats = []
    row_meta = []
    for idx in indices:
        row = rows[idx]
        image = images[int(row["image_id"])]
        path = FRAMES / str(image["file_name"])
        with Image.open(path) as raw:
            rgb = raw.convert("RGB")
            crop = crop_box(rgb, [float(v) for v in row["bbox"]])
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            pixel_stats.append({"mean": float(arr.mean()), "std": float(arr.std()), "shape": list(arr.shape), "path": str(path)})
            tensors.append(tf(crop).unsqueeze(0))
        row_meta.append({"index": idx, "image_id": int(row["image_id"]), "video_id": int(row["video_id"]), "frame_index": int(image.get("frame_index", -1))})
    x = torch.cat(tensors, dim=0).to(args.device)
    with torch.no_grad():
        fresh = torch.nn.functional.normalize(model.forward_features(x)["x_norm_clstoken"], dim=-1).cpu().numpy().astype(np.float32)
    entries = []
    for j, idx in enumerate(indices):
        c = cached[idx]
        f = fresh[j]
        entries.append({
            **row_meta[j],
            "pixel": pixel_stats[j],
            "fresh_norm": float(np.linalg.norm(f)),
            "cache_norm": float(np.linalg.norm(c)),
            "fresh_cache_cosine": cosine(f, c),
            "fresh_cache_max_abs_diff": float(np.max(np.abs(f - c))),
            "fresh_vector_sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "cache_vector_sha256": hashlib.sha256(c.tobytes()).hexdigest(),
        })
    pairwise = []
    for i, a in enumerate(fresh):
        for j in range(i + 1, len(fresh)):
            pairwise.append({"i": indices[i], "j": indices[j], "fresh_cosine": cosine(a, fresh[j]), "cache_cosine": cosine(cached[indices[i]], cached[indices[j]])})
    out = {
        "schema_version": "trackocd.phase82r.appearance_cache_audit.v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "q0_path": str(Q0), "q0_sha256": sha256(Q0), "cache_path": str(CACHE), "cache_sha256": sha256(CACHE),
        "model_repo": str(HUB), "model": "facebookresearch/dinov2:dinov2_vitb14", "device": args.device,
        "indices": indices, "entries": entries, "pairwise": pairwise,
        "forbidden_inference_fields": ["category_id", "track_id", "physical_id", "semantic_id", "future", "held_gt", "text"],
        "public_dev_q1_sealed_accessed": False,
        "interpretation": "fresh direct RGB crop inference compared with cached vectors; diagnostic only",
    }
    atomic_json(args.out, out)
    print(json.dumps({"out": str(args.out), "entries": entries, "pairwise": pairwise}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
