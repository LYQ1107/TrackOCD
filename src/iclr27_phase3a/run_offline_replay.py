#!/usr/bin/env python3
"""Offline replay of the exported SimOWT pre-association packages.

Reconstructs the per-image final trajectory JSON exactly like
`IDOL.track_eval` does online, but without reading images, running the
detector, or using any external data beyond the exported packages.
"""
from __future__ import annotations

import json
import importlib.util
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
SIMOWT_ROOT = PROJECT_ROOT / "third_party" / "SimOWT"
sys.path.insert(0, str(SIMOWT_ROOT))
sys.path.insert(0, str(SIMOWT_ROOT / "projects"))
sys.path.insert(0, str(SIMOWT_ROOT / "detectron2"))

_tracker_path = SIMOWT_ROOT / "projects" / "IDOL" / "idol" / "models" / "tracker.py"
_spec = importlib.util.spec_from_file_location("simowt_tracker", _tracker_path)
_tracker_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tracker_mod)
IDOL_Tracker = _tracker_mod.IDOL_Tracker

EXPORT_DIR = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "smoke"
OUT_DIR = PROJECT_ROOT / "outputs" / "iclr27_phase3a" / "trajectories" / "offline_replay_20"


def resize_output_shape(oldh: int, oldw: int, short_edge_length: int = 640, max_size: int = 1333):
    """Exact replica of detectron2 ResizeShortestEdge.get_output_shape."""
    h, w = oldh, oldw
    size = short_edge_length * 1.0
    scale = size / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    return int(newh + 0.5), int(neww + 0.5)


def load_tracker_config(video_id: int) -> dict:
    cfg_path = EXPORT_DIR / "replay_packages" / str(video_id) / "tracker_config.json"
    with open(cfg_path) as f:
        return json.load(f)


def build_tracker(cfg: dict) -> IDOL_Tracker:
    return IDOL_Tracker(
        init_score_thr=cfg["init_score_thr"],
        obj_score_thr=cfg["obj_score_thr"],
        nms_thr_pre=cfg["nms_thr_pre"],
        nms_thr_post=cfg.get("nms_thr_post", 0.05),
        addnew_score_thr=cfg["addnew_score_thr"],
        memo_tracklet_frames=cfg["memo_tracklet_frames"],
        memo_momentum=cfg["memo_momentum"],
        long_match=cfg["long_match"],
        frame_weight=cfg["frame_weight"],
        temporal_weight=cfg["temporal_weight"],
        memory_len=cfg["memory_len"],
        match_score_thr=cfg.get("match_score_thr", 0.5),
    )


def replay_video(video_id: int, frames: list[dict], device: torch.device) -> int:
    pkg_dir = EXPORT_DIR / "replay_packages" / str(video_id)
    cfg = load_tracker_config(video_id)
    tracker = build_tracker(cfg)
    n_written = 0
    for fi, frame_meta in enumerate(frames):
        npz_path = pkg_dir / f"frame_{fi:06d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        z = np.load(npz_path)

        bboxes = torch.from_numpy(z["det_bboxes"]).to(device)
        labels = torch.from_numpy(z["det_labels"]).to(device)
        masks = torch.from_numpy(z["det_masks"]).float().to(device)
        track_feats = torch.from_numpy(z["track_feats"]).float().to(device)
        indices = torch.from_numpy(z["indices"]).to(device)
        frame_id = int(z["frame_id"])
        ori_h, ori_w = [int(x) for x in z["ori_size"]]
        if "image_size" in z.files:
            image_size = (int(z["image_size"][0]), int(z["image_size"][1]))
        else:
            image_size = resize_output_shape(ori_h, ori_w)

        bboxes_out, labels_out, ids, indices_out, masks_after_track = tracker.match(
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            track_feats=track_feats,
            frame_id=frame_id,
            indices=indices,
            datasets_ori=[],
            ori_size=(ori_h, ori_w),
        )
        del bboxes, labels, masks, track_feats, indices

        selected = torch.nonzero(ids >= 0).squeeze(-1)
        image_id = frame_meta["image_id"]
        preds = []
        if selected.shape[0] != 0:
            N, C, H, W = masks_after_track.shape
            mask = F.interpolate(
                masks_after_track,
                size=(H * 4, W * 4),
                mode="bilinear",
                align_corners=False,
            )
            mask = mask.sigmoid() > 0.5
            masks_after_track1 = mask[:, :, : image_size[0], : image_size[1]]
            mask = F.interpolate(
                masks_after_track1.float(),
                size=(ori_h, ori_w),
                mode="nearest",
            )
            mask = mask.squeeze(1).byte()
            masks_after_track2 = mask
            rles = [
                mask_util.encode(np.array(mask1[:, :, None], order="F", dtype="uint8"))[0]
                for mask1 in masks_after_track2.cpu()
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            for idx in selected.tolist():
                det = bboxes_out[idx]
                mask_rle = rles[idx]
                x1n, y1n, x2n, y2n = det[:4].tolist()
                w = int(x2n * ori_w)
                h = int(y2n * ori_h)
                c_x = int(x1n * ori_w)
                c_y = int(y1n * ori_h)
                x1 = max(0, int(c_x - w / 2))
                y1 = max(0, int(c_y - h / 2))
                x2 = int(c_x + w / 2)
                y2 = int(c_y + h / 2)
                preds.append(
                    {
                        "bbox": [x1, y1, w, h],
                        "track_id": int(ids[idx].item()),
                        "category_id": 1,
                        "image_id": image_id,
                        "video_id": video_id,
                        "score": float(det[-1].item()),
                        "segmentations": mask_rle,
                    }
                )

        target = OUT_DIR / f"{str(image_id).zfill(10)}.json"
        fd, tmp = tempfile.mkstemp(prefix="replay_", suffix=".json", dir=OUT_DIR)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(preds, f, separators=(",", ":"))
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        n_written += 1
    return n_written


def main() -> None:
    frames_by_video: dict[int, list[dict]] = defaultdict(list)
    for det_file in (EXPORT_DIR / "pre_assoc_detections").glob("*.jsonl"):
        video_id = int(det_file.stem)
        seen = set()
        for line in det_file.read_text().splitlines():
            r = json.loads(line)
            fo = r["frame_order"]
            if fo in seen:
                continue
            seen.add(fo)
            frames_by_video[video_id].append(
                {"frame_order": fo, "image_id": r["image_id"]}
            )
        frames_by_video[video_id].sort(key=lambda x: x["frame_order"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_id in sorted(frames_by_video):
        n = replay_video(video_id, frames_by_video[video_id], device)
        total += n
        print("replayed", video_id, n, "frames", flush=True)
    print("total frames", total)


if __name__ == "__main__":
    main()
