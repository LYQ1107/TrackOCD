"""Frozen DINOv2 single-frame features for Phase 4L held-out detections.

Same crop/transform protocol as Phase 4I
(src/iclr27_phase4i/extract_detection_features.py), parameterized for the
held-out export.  Online-legal current-frame evidence only.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.data.tao_io import FRAMES_ROOT
from src.features.extract import (
    crop_bbox,
    load_dinov2,
    make_crop_transform,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-root", type=Path, required=True)
    ap.add_argument("--image-json", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--video-ids", default="")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    model, size = load_dinov2()
    transform = make_crop_transform(size)
    d = json.loads(args.image_json.read_text())
    image_map = {im["id"]: im for im in d["images"]}
    video_ids = [int(v) for v in args.video_ids.split(",") if v.strip()]
    if not video_ids:
        video_ids = sorted(int(p.stem) for p in
                           args.pre_root.glob("*.jsonl"))

    torch.cuda.set_device(args.device)
    model = model.cuda(args.device).eval()
    for vid in video_ids:
        out = args.out_root / str(vid) / "feats.npz"
        if out.exists():
            print("skip", vid, flush=True)
            continue
        dets = [json.loads(l) for l in
                (args.pre_root / f"{vid}.jsonl").read_text().splitlines()
                if l.strip()]
        by_frame = defaultdict(list)
        for i, r in enumerate(dets):
            by_frame[int(r["frame_order"])].append((i, r))
        feats = np.zeros((len(dets), 768), dtype=np.float16)
        frame_orders = np.zeros(len(dets), dtype=np.int64)
        det_ids = np.zeros(len(dets), dtype=np.int64)
        scores = np.zeros(len(dets), dtype=np.float32)
        for i, r in enumerate(dets):
            frame_orders[i] = int(r["frame_order"])
            det_ids[i] = int(r["det_local_id"])
            scores[i] = float(r["score"])
        n_ok = 0
        for fo in sorted(by_frame):
            entries = by_frame[fo]
            tensors, valid = [], []
            for i, r in entries:
                im = image_map.get(int(r["image_id"]))
                if im is None:
                    continue
                p = FRAMES_ROOT / im["file_name"]
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as exc:
                    print("IMGERR", vid, r["image_id"], exc, flush=True)
                    continue
                crop = crop_bbox(img, r["bbox_xyxy_original"])
                if min(crop.size) < 4:
                    continue
                tensors.append(transform(crop).unsqueeze(0))
                valid.append((i, r))
            if not valid:
                continue
            for s in range(0, len(tensors), args.batch):
                chunk = torch.cat(tensors[s:s + args.batch], dim=0).cuda(
                    args.device)
                with torch.no_grad():
                    f = model(chunk)
                    f = torch.nn.functional.normalize(f, dim=-1)
                f = f.float().cpu().numpy().astype(np.float16)
                for k, (i, r) in enumerate(valid[s:s + args.batch]):
                    feats[i] = f[k]
                    n_ok += 1
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, feats=feats, frame_orders=frame_orders,
                 det_local_ids=det_ids, scores=scores)
        print("video", vid, "dets", len(dets), "ok", n_ok,
              "frames", len(by_frame), flush=True)


if __name__ == "__main__":
    main()
