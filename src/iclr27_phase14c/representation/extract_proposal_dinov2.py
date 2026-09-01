"""Causal DINOv2 proposal-crop extraction for Phase14C."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/iclr27_phase14c/sources/tao_train_frames"


def crop_bbox(img, box, context=0.10):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    x1c, y1c = max(0.0, cx - nw / 2), max(0.0, cy - nh / 2)
    x2c, y2c = min(float(w), cx + nw / 2), min(float(h), cy + nh / 2)
    if x2c - x1c < 2 or y2c - y1c < 2:
        x1c, y1c, x2c, y2c = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    return img.crop((int(x1c), int(y1c), int(x2c), int(y2c)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="outputs/iclr27_phase14c/proposals/proposals_mixed.csv")
    ap.add_argument("--annotation", default="data/iclr27_phase14c/manifests/phase14c_validation_train.json")
    ap.add_argument("--out", default="outputs/iclr27_phase14c/features/proposal_dinov2.npz")
    ap.add_argument("--meta", default="outputs/iclr27_phase14c/features/proposal_dinov2_meta.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / args.proposals).open()))
    ann = json.loads((ROOT / args.annotation).read_text())
    imgs = {int(x["id"]): x for x in ann["images"]}
    tf = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval().to(args.device)
    feats = np.zeros((len(rows), 768), dtype=np.float32)
    failures = []
    tensors, indices = [], []

    def flush():
        nonlocal tensors, indices
        if not tensors:
            return
        batch = torch.cat(tensors, dim=0).to(args.device)
        with torch.no_grad():
            f = torch.nn.functional.normalize(model(batch), dim=-1).cpu().numpy().astype(np.float32)
        for j, i in enumerate(indices):
            feats[i] = f[j]
        tensors, indices = [], []

    for i, r in enumerate(rows):
        im = imgs[int(r["image_id"])]
        path = FRAMES / im["file_name"]
        try:
            with Image.open(path) as x:
                crop = crop_bbox(x.convert("RGB"), json.loads(r["bbox_xyxy"]))
            tensors.append(tf(crop).unsqueeze(0)); indices.append(i)
            if len(tensors) >= args.batch:
                flush()
        except Exception as e:
            failures.append({"row": i, "image_id": int(r["image_id"]), "path": str(path), "error": repr(e)})
    flush()
    assert not failures, failures[:2]
    assert np.isfinite(feats).all() and np.all(np.linalg.norm(feats, axis=1) > 0.99)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    np.savez_compressed(tmp, feats=feats, row_keys=np.asarray([
        f"{r['video_id']}:{r['frame_id']}:{r['proposal_local_id']}:{r['track_id']}" for r in rows]))
    # np.savez appends .npz when the suffix is not .npz; use the generated path.
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
    os.replace(generated, out)
    meta = {"rows": len(rows), "shape": list(feats.shape), "failures": failures,
            "source_proposals": str((ROOT / args.proposals).resolve()),
            "source_frames": str(FRAMES.resolve()), "context": 0.10,
            "future_frames_used": False, "q1_label_used": False,
            "private_gt_used": False, "physical_id_used_as_feature": False}
    mp = ROOT / args.meta; mp.parent.mkdir(parents=True, exist_ok=True)
    mt = mp.with_suffix(mp.suffix + ".tmp"); mt.write_text(json.dumps(meta, indent=2)); os.replace(mt, mp)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
