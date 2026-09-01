"""Extract DINOv2 ViT-B/14 bbox-crop features for the frozen Q1/Q2 dev streams.

Same recipe as src/features/extract.py (518px, context=0.10) so the dev
features are compatible with the train-known feature cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from src.iclr27_phase4s.protocol import TAO_VAL_ANN, load_proposals

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data" / "raw" / "tao" / "frames"


def load_model(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(device)
    return model


def transform_crop():
    return transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def crop_bbox(img, box, context=0.10):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    x1c, y1c = max(0, cx - nw / 2), max(0, cy - nh / 2)
    x2c, y2c = min(w, cx + nw / 2), min(h, cy + nh / 2)
    if x2c - x1c < 2 or y2c - y1c < 2:
        x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = load_proposals(Path(args.proposals))
    if args.limit:
        rows = rows[: args.limit]
    val = json.loads(TAO_VAL_ANN.read_text())
    img_map = {im["id"]: im for im in val["images"]}
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.device)
    tf = transform_crop()

    by_img = {}
    for i, r in enumerate(rows):
        im = img_map[int(r["image_id"])]
        by_img.setdefault(im["file_name"], []).append(i)

    feats = [None] * len(rows)
    cache = {}
    order = list(by_img)
    tensors = []
    idx_buf = []
    for img_path in order:
        try:
            img = Image.open(FRAMES / img_path).convert("RGB")
        except Exception:
            for i in by_img[img_path]:
                feats[i] = np.zeros(768, dtype=np.float32)
            continue
        cache[img_path] = img
        for i in by_img[img_path]:
            b = json.loads(rows[i]["bbox_xyxy"])
            t = tf(crop_bbox(img, b)).unsqueeze(0)
            tensors.append(t)
            idx_buf.append(i)
            if len(tensors) >= args.batch:
                batch = torch.cat(tensors, dim=0).to(args.device)
                with torch.no_grad():
                    f = torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy()
                for j, ii in enumerate(idx_buf):
                    feats[ii] = f[j].astype(np.float32)
                tensors, idx_buf = [], []
    if tensors:
        batch = torch.cat(tensors, dim=0).to(args.device)
        with torch.no_grad():
            f = torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy()
        for j, ii in enumerate(idx_buf):
            feats[ii] = f[j].astype(np.float32)

    arr = (np.stack([x for x in feats]).astype(np.float32)
           if feats else np.zeros((0, 768), dtype=np.float32))
    np.savez_compressed(out / "feats.npz", feats=arr)
    meta = {"n_rows": len(rows), "shape": list(arr.shape), "proposals": args.proposals}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved", out / "feats.npz", arr.shape)


if __name__ == "__main__":
    main()
