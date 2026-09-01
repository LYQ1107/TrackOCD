"""Extract frozen DINOv3 object-crop features for the locked Q1 stream.

This is a feature-only audit asset.  It does not read private GT labels and
does not alter the physical stream or semantic decision process.  Cropping,
sampling, and normalization match the locally audited DINOv3 GT-track cache.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/raw/tao/frames"
ANN = ROOT / "data/raw/tao/annotations/validation.json"


def crop_bbox(img: Image.Image, box, context: float = 0.10) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    nw, nh = bw * (1.0 + 2.0 * context), bh * (1.0 + 2.0 * context)
    x1c, y1c = max(0.0, cx - nw * 0.5), max(0.0, cy - nh * 0.5)
    x2c, y2c = min(float(w), cx + nw * 0.5), min(float(h), cy + nh * 0.5)
    if x2c - x1c < 2.0 or y2c - y1c < 2.0:
        x1c, y1c = max(0.0, x1), max(0.0, y1)
        x2c, y2c = min(float(w), x2), min(float(h), y2)
    return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:2")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # Import only after argument parsing so a CPU smoke test can inspect the
    # protocol without constructing the foundation model.
    from src.dinov3_bakeoff.adapter import DinoV3Adapter

    with (ROOT / args.proposals).open(newline="") as f:
        rows = list(csv.DictReader(f))
    ann = json.loads(ANN.read_text())
    image_by_id = {int(x["id"]): x for x in ann["images"]}
    tf = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
    adapter = DinoV3Adapter(device=args.device, feature_mode="cls")
    feats = [None] * len(rows)
    tensors, indices = [], []
    failures = []

    def flush() -> None:
        nonlocal tensors, indices
        if not tensors:
            return
        with torch.no_grad():
            out = adapter.embed_crops(tensors)
        for j, idx in enumerate(indices):
            feats[idx] = out[j]
        tensors, indices = [], []

    for i, row in enumerate(rows):
        image = image_by_id.get(int(row["image_id"]))
        if image is None:
            failures.append({"row": i, "reason": "missing_image_record"})
            feats[i] = np.zeros(adapter.feature_dim, dtype=np.float32)
            continue
        path = FRAMES / image["file_name"]
        try:
            with Image.open(path) as im:
                crop = crop_bbox(im.convert("RGB"), json.loads(row["bbox_xyxy"]))
            # Match the frozen DINOv2 extractor exactly: even a zero-width
            # clamped crop is passed through PIL resize, so FP rows are not
            # silently removed from the locked stream.
            tensors.append(tf(crop))
            indices.append(i)
            if len(tensors) >= args.batch:
                flush()
        except Exception as exc:  # preserve row count; report, do not skip
            failures.append({"row": i, "reason": type(exc).__name__})
            feats[i] = np.zeros(adapter.feature_dim, dtype=np.float32)
    flush()

    if any(x is None for x in feats):
        raise RuntimeError("feature extraction left an unset row")
    arr = np.asarray(feats, dtype=np.float32)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    np.savez_compressed(tmp, feats=arr)
    # np.savez appends .npz when passed a suffix-less temp path; normalize it.
    tmp_npz = Path(str(tmp) + ".npz") if not tmp.exists() else tmp
    tmp_npz.replace(out)
    meta = {
        "representation": "DINOv3 ViT-B/16 LVD-1689M CLS, frozen",
        "weight_source": "timm converted distribution (W4)",
        "weight_sha256": "1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332",
        "proposals": str((ROOT / args.proposals).resolve()),
        "rows": len(rows),
        "shape": list(arr.shape),
        "private_gt_used": False,
        "q1_labels_used": False,
        "future_used": False,
        "physical_id_used": False,
        "failed_rows": failures,
    }
    (out.parent / "q1_dinov3_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
