"""DINOv2 ViT-B/14 bbox-crop features for the real TAO-train tracker stream.
Same recipe as src/features/extract.py (518px, context=0.10)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-root", default="data/raw/tao/frames")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    import csv
    rows = list(csv.DictReader(open(ROOT / args.proposals)))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(args.device)
    tf = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    frames = ROOT / args.frames_root

    def crop(img, box, ctx=0.10):
        w, h = img.size
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nw, nh = bw * (1 + 2 * ctx), bh * (1 + 2 * ctx)
        x1c, y1c = max(0, cx - nw / 2), max(0, cy - nh / 2)
        x2c, y2c = min(w, cx + nw / 2), min(h, cy + nh / 2)
        if x2c - x1c < 2 or y2c - y1c < 2:
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))

    train = json.loads((ROOT / "data" / "raw" / "tao" / "annotations" / "train.json").read_text())
    img_map = {im["id"]: im for im in train["images"]}
    feats = [None] * len(rows)
    by_img = {}
    for i, r in enumerate(rows):
        by_img.setdefault(int(r["image_id"]), []).append(i)
    tensors, idx_buf = [], []
    for img_id, idxs in sorted(by_img.items()):
        im = img_map.get(img_id)
        if im is None:
            continue
        try:
            img = Image.open(frames / im["file_name"]).convert("RGB")
        except Exception:
            continue
        for i in idxs:
            b = json.loads(rows[i]["bbox_xyxy"])
            tensors.append(tf(crop(img, b)).unsqueeze(0))
            idx_buf.append(i)
            if len(tensors) >= args.batch:
                batch = torch.cat(tensors).to(args.device)
                with torch.no_grad():
                    f = torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy()
                for j, ii in enumerate(idx_buf):
                    feats[ii] = f[j].astype(np.float32)
                tensors, idx_buf = [], []
    if tensors:
        batch = torch.cat(tensors).to(args.device)
        with torch.no_grad():
            f = torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy()
        for j, ii in enumerate(idx_buf):
            feats[ii] = f[j].astype(np.float32)
    arr = np.stack([x if x is not None else np.zeros(768, dtype=np.float32) for x in feats])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "feats.npz", feats=arr)
    print("saved", out / "feats.npz", arr.shape, "nonzero", int(np.count_nonzero(arr.any(axis=1))))


if __name__ == "__main__":
    main()
