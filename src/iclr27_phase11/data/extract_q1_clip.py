"""Extract cached OpenAI CLIP frame features for the corrected Q1 proposals.

This is a feature-only pass over public proposal crops.  It reads no Q1
labels and writes an atomic row-aligned ``feats.npz`` for the representation
prototype.  The CLIP visual encoder is pretrained with language alignment;
the later trajectory adapter is the only trainable part.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ANN = ROOT / "data/raw/tao/annotations/validation.json"
PROPOSALS = ROOT / "outputs/iclr27_phase6b/q1/final_dsct/proposals_dev.csv"
OUT = ROOT / "outputs/iclr27_phase11/assets/q1_clip_feats.npz"


def crop(im: Image.Image, box, context=0.10):
    w, h = im.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = bw * (1.0 + 2.0 * context), bh * (1.0 + 2.0 * context)
    x1 = max(0.0, cx - nw / 2.0); y1 = max(0.0, cy - nh / 2.0)
    x2 = min(float(w), cx + nw / 2.0); y2 = min(float(h), cy + nh / 2.0)
    return im.crop((int(x1), int(y1), int(x2), int(y2)))


def atomic_npz(path: Path, feats: np.ndarray, image_ids: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, feats=feats.astype(np.float32), image_ids=image_ids)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default=str(PROPOSALS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device)
    model.eval()
    ann = json.loads(ANN.read_text())
    image_by_id = {int(im["id"]): im for im in ann["images"]}
    rows = []
    with open(args.proposals, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    feats = []
    image_ids = []
    for start in range(0, len(rows), args.batch_size):
        batch = []
        bid = []
        for r in rows[start:start + args.batch_size]:
            image_id = int(r["image_id"])
            imrec = image_by_id[image_id]
            path = ROOT / "data/raw/tao/frames" / imrec["file_name"]
            with Image.open(path).convert("RGB") as im:
                box = json.loads(r["bbox_xyxy"])
                batch.append(preprocess(crop(im, box)))
            bid.append(image_id)
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            z = model.encode_image(x)
            z = torch.nn.functional.normalize(z, dim=-1)
        feats.append(z.cpu().numpy().astype(np.float32))
        image_ids.extend(bid)
        print(f"encoded {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    arr = np.concatenate(feats, axis=0)
    if len(arr) != len(rows):
        raise RuntimeError(f"feature/row mismatch {len(arr)} vs {len(rows)}")
    atomic_npz(Path(args.out), arr, np.asarray(image_ids, dtype=np.int64))
    meta = {
        "rows": len(rows), "dim": int(arr.shape[1]),
        "encoder": "OpenAI CLIP ViT-B/32 visual encoder",
        "proposals": str(Path(args.proposals).resolve()),
        "labels_used": False, "future_used": False,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
