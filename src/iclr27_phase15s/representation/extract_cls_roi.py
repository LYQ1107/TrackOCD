"""Matched frozen DINOv2 CLS and proposal-interior ROI patch features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/iclr27_phase15s/sources/tao_train_frames"


def l2(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def crop_box(img, box, context=0.10):
    w, h = img.size; x1, y1, x2, y2 = map(float, box); bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    xa, ya, xb, yb = max(0, cx - nw / 2), max(0, cy - nh / 2), min(w, cx + nw / 2), min(h, cy + nh / 2)
    if xb - xa < 2 or yb - ya < 2: xa, ya, xb, yb = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    return img.crop((int(xa), int(ya), int(xb), int(yb))), (xa, ya, xb, yb)


def row_key(r):
    return f"{r.get('video_id')}:{r.get('frame_id')}:{r.get('proposal_local_id')}:{r.get('track_id')}:{r.get('image_id')}"


def legacy_row_key(r):
    return f"{r.get('video_id')}:{r.get('frame_id')}:{r.get('proposal_local_id')}:{r.get('track_id')}"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).resolve().open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def extract(proposals: Path, annotation: Path, out: Path, device: str, batch: int, context: float = 0.10, reuse_cls: Path | None = None):
    rows = list(csv.DictReader(proposals.open())); ann = json.loads(annotation.read_text()); imgs = {int(x["id"]): x for x in ann["images"]}
    tf = transforms.Compose([transforms.Resize((518, 518), interpolation=Image.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(device)
    cls = np.zeros((len(rows), 768), np.float32); roi = np.zeros_like(cls); failures = []
    reused = None
    if reuse_cls is not None:
        z = np.load(reuse_cls, allow_pickle=False)
        if "feats" not in z or "row_keys" not in z: raise RuntimeError("reuse CLS cache lacks feats/row_keys")
        keys = np.asarray([row_key(r) for r in rows]); legacy = np.asarray([legacy_row_key(r) for r in rows])
        cached = z["row_keys"].astype(str)
        if len(z["feats"]) != len(rows) or not (np.array_equal(cached, keys.astype(str)) or np.array_equal(cached, legacy.astype(str))):
            raise RuntimeError("reuse CLS cache row order/key mismatch")
        reused = np.asarray(z["feats"], dtype=np.float32); cls[:] = reused
    tensors, masks, idxs = [], [], []

    def flush():
        nonlocal tensors, masks, idxs
        if not tensors: return
        x = torch.cat(tensors, 0).to(device)
        with torch.no_grad():
            ff = model.forward_features(x)
            cc = torch.nn.functional.normalize(ff["x_norm_clstoken"], dim=-1).cpu().numpy()
            pp = ff["x_norm_patchtokens"].reshape(len(idxs), 37, 37, 768).cpu().numpy()
        for j, i in enumerate(idxs):
            if reused is None: cls[i] = cc[j]
            q = pp[j][masks[j]]
            if q.shape[0] == 0: q = pp[j].reshape(-1, 768)
            roi[i] = l2(q.mean(axis=0))
        tensors, masks, idxs = [], [], []

    for i, r in enumerate(rows):
        try:
            im = imgs[int(r["image_id"])]
            path = FRAMES / im["file_name"]
            with Image.open(path) as raw: crop, cb = crop_box(raw.convert("RGB"), json.loads(r["bbox_xyxy"]))
            xa, ya, xb, yb = cb; box = json.loads(r["bbox_xyxy"]); sx, sy = 518 / max(xb - xa, 1e-6), 518 / max(yb - ya, 1e-6)
            rx1, ry1, rx2, ry2 = (box[0] - xa) * sx, (box[1] - ya) * sy, (box[2] - xa) * sx, (box[3] - ya) * sy
            centers = (np.arange(37, dtype=np.float32) + 0.5) * 14.0
            mask = ((centers[None, :] >= rx1) & (centers[None, :] <= rx2) & (centers[:, None] >= ry1) & (centers[:, None] <= ry2))
            tensors.append(tf(crop).unsqueeze(0)); masks.append(mask); idxs.append(i)
            if len(tensors) >= batch: flush()
        except Exception as e:
            failures.append({"row": i, "key": row_key(r), "error": repr(e)})
    flush()
    if failures: raise RuntimeError(f"feature extraction failures: {failures[:2]}")
    if not np.isfinite(cls).all() or not np.isfinite(roi).all() or np.any(np.linalg.norm(roi, axis=1) < 0.99): raise RuntimeError("non-finite or non-normalized feature")
    out.parent.mkdir(parents=True, exist_ok=True); tmp = Path(str(out) + ".tmp")
    np.savez_compressed(tmp, cls=cls, roi=roi, row_keys=np.asarray([row_key(r) for r in rows]))
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp; os.replace(generated, out)
    meta = {"protocol": "trackocd_iclr27_phase15s16", "rows": len(rows), "shape": list(cls.shape), "context": context, "source_proposals": str(proposals.resolve()), "source_annotation": str(annotation.resolve()), "source_frames": str(FRAMES.resolve()), "proposal_sha256": sha256(proposals), "future_frames_used": False, "q1_label_used": False, "devplus_labels_for_fit": False, "physical_id_used_as_feature": False, "failures": failures, "feature_names": ["normalized_cls", "normalized_roi_patch_mean"], "device": device, "batch": batch, "cls_reused_from": str(reuse_cls.resolve()) if reuse_cls is not None else None}
    mp = Path(str(out) + ".json"); mt = Path(str(mp) + ".tmp"); mt.write_text(json.dumps(meta, indent=2, sort_keys=True)); os.replace(mt, mp)
    print(json.dumps(meta, indent=2, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--proposals", required=True); ap.add_argument("--annotation", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:2"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--reuse-cls"); args = ap.parse_args()
    extract(ROOT / args.proposals, ROOT / args.annotation, ROOT / args.out, args.device, args.batch, reuse_cls=(ROOT / args.reuse_cls if args.reuse_cls else None))


if __name__ == "__main__": main()
