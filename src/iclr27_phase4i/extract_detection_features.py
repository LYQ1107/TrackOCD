"""Extract frozen DINOv2 single-frame features for pre-association
detections of the Phase 3A 20-video subset.

Every detection is cropped with the same protocol as the Phase 4F feature
cache (bbox context 0.10, resize 518, ImageNet normalization, DINOv2 ViT-B/14)
and stored per video as float16 arrays aligned with the replay npz row order.
This is the legal current-frame semantic evidence input: no future frames,
no complete-track features, no GT.
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

PRE = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "pre_assoc_detections"
TAO_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
OUT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "detection_features"


def load_image_map():
    d = json.loads(TAO_JSON.read_text())
    return {im["id"]: im for im in d["images"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-ids", default="")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    model, size = load_dinov2()
    transform = make_crop_transform(size)
    image_map = load_image_map()

    video_ids = [int(v) for v in args.video_ids.split(",") if v.strip()]
    if not video_ids:
        video_ids = sorted(int(p.stem) for p in PRE.glob("*.jsonl"))

    for vid in video_ids:
        out = OUT_ROOT / str(vid) / "feats.npz"
        if out.exists():
            print("skip", vid, flush=True)
            continue
        dets = [json.loads(l) for l in (PRE / f"{vid}.jsonl").read_text().splitlines() if l.strip()]
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
            tensors = []
            valid = []
            for i, r in entries:
                im = image_map.get(int(r["image_id"]))
                if im is None:
                    continue
                p = FRAMES_ROOT / im["file_name"]
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as e:
                    print("IMGERR", vid, r["image_id"], e, flush=True)
                    continue
                crop = crop_bbox(img, r["bbox_xyxy_original"])
                if min(crop.size) < 4:
                    continue
                tensors.append(transform(crop).unsqueeze(0).cuda())
                valid.append((i, r))
            if not valid:
                continue
            for s in range(0, len(tensors), args.batch):
                chunk = torch.cat(tensors[s:s + args.batch], dim=0)
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
