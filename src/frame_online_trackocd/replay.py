"""Frame-online replay driver for Phase 4I.

Modes:
  B0: original IDOL association, no semantics (equivalence target).
  B1: original association, then frame-online semantics on predicted tracks.
  B2: semantic observations before association + soft semantic cost in the
      score matrix, then semantic state updates after association.

The output JSON format and mask post-processing replicate Phase 3A exactly
so B0 can be compared byte-for-byte with the original/instrumented runs.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")

from src.frame_online_trackocd.frame_tracker import FrameOnlineTracker


def resize_output_shape(oldh, oldw, short_edge_length=640, max_size=1333):
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


def load_tracker_config(video_id, export_root):
    p = export_root / "replay_packages" / str(video_id) / "tracker_config.json"
    with open(p) as f:
        return json.load(f)


def build_tracker(cfg, association_logger=None):
    return FrameOnlineTracker(
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
        association_logger=association_logger,
    )


def _load_frame_feats(feat_root, video_id, frame_id):
    arr = np.load(feat_root / str(video_id) / "feats.npz")
    fo = arr["frame_orders"]
    idx = np.where(fo == int(frame_id))[0]
    return arr["feats"][idx].astype(np.float32), idx


def replay_video(video_id, frames, export_root, feat_root, out_dir,
                 sem_manager=None, mode="B0", lambda_s=0.0,
                 device=None, log_writer=None, association_logger=None):
    cfg = load_tracker_config(video_id, export_root)
    tracker = build_tracker(cfg, association_logger=association_logger)
    tracker.video_id = video_id
    pkg_dir = export_root / "replay_packages" / str(video_id)
    log_rows = []
    for fi, frame_meta in enumerate(frames):
        npz_path = pkg_dir / f"frame_{fi:06d}.npz"
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

        sem_cost = None
        det_obs = None
        if mode in ("B1", "B2") and sem_manager is not None:
            sem_manager.video_id = video_id
            sem_manager.current_frame = frame_id
            feats, feat_idx = _load_frame_feats(feat_root, video_id, frame_id)
            assert feats.shape[0] == bboxes.shape[0], (
                video_id, frame_id, feats.shape[0], bboxes.shape[0])
            det_obs = sem_manager.observe(feats)
            if mode == "B2":
                active_ids = []
                if not tracker.empty:
                    memo_ids = tracker.memo[3]
                    active_ids = [int(x) for x in memo_ids.tolist()]
                sem_cost = sem_manager.semantic_cost_matrix(
                    det_obs, active_ids)
                if sem_cost is not None:
                    sem_cost = torch.from_numpy(sem_cost).to(device)

        bboxes_out, labels_out, ids, indices_out, masks_after_track = \
            tracker.match(
                bboxes=bboxes, labels=labels, masks=masks,
                track_feats=track_feats, frame_id=frame_id,
                indices=indices, datasets_ori=[],
                ori_size=(ori_h, ori_w),
                sem_cost=sem_cost, lambda_s=lambda_s)

        if mode in ("B1", "B2") and sem_manager is not None:
            if association_logger is not None:
                association_logger.finish_frame(sem_manager, det_obs,
                                                lambda_s)
            scores = bboxes[:, 4].cpu().numpy()
            bbox_xy = bboxes[:, :4].cpu().numpy()
            # ids are aligned post mask-nms; map to raw npz det rows via the
            # tracker's recorded survivor positions
            raw_idx = getattr(tracker, "last_raw_positions",
                              list(range(len(ids))))
            mask_ok = (torch.sigmoid(masks) > 0.5).float().mean(
                dim=(1, 2, 3)).cpu().numpy()
            app_norm = torch.norm(track_feats, dim=1).cpu().numpy()
            ap_map = {}
            if association_logger is not None:
                for r in association_logger.rows:
                    if int(r["frame_id"]) == int(frame_id):
                        ap_map[int(r["raw_det_idx"])] = float(
                            r["appearance_best_score"])
            det_aux = [{
                "det_score": float(bboxes[i, 4]),
                "mask_area_frac": float(mask_ok[i]),
                "appearance_norm": float(app_norm[i]),
                "assoc_ap_score": ap_map.get(i, 0.0),
            } for i in range(len(bboxes))]
            sem_manager.post_association_raw(
                frame_id, ids, bbox_xy, scores, det_obs, raw_idx,
                det_aux=det_aux)
            if log_writer is not None:
                for p, ridx in enumerate(raw_idx):
                    tid = int(ids[p]) if p < len(ids) else -2
                    cx, cy, w, h = bbox_xy[ridx]
                    pix = [float((cx - w / 2) * ori_w),
                           float((cy - h / 2) * ori_h),
                           float((cx + w / 2) * ori_w),
                           float((cy + h / 2) * ori_h)]
                    row = sem_manager.log_row(frame_id, ridx, det_obs, tid,
                                              scores[ridx], pix)
                    row["video_id"] = video_id
                    row["image_id"] = int(frame_meta["image_id"])
                    log_rows.append(row)

        selected = torch.nonzero(ids >= 0).squeeze(-1)
        image_id = frame_meta["image_id"]
        preds = []
        if selected.shape[0] != 0:
            N, C, H, W = masks_after_track.shape
            mask = F.interpolate(masks_after_track, size=(H * 4, W * 4),
                                 mode="bilinear", align_corners=False)
            mask = mask.sigmoid() > 0.5
            masks_after_track1 = mask[:, :, : image_size[0], : image_size[1]]
            mask = F.interpolate(masks_after_track1.float(),
                                 size=(ori_h, ori_w), mode="nearest")
            mask = mask.squeeze(1).byte()
            rles = [mask_util.encode(np.array(m1[:, :, None], order="F",
                                              dtype="uint8"))[0]
                    for m1 in mask.cpu()]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            for idx in selected.tolist():
                det = bboxes_out[idx]
                x1n, y1n, x2n, y2n = det[:4].tolist()
                w = int(x2n * ori_w)
                h = int(y2n * ori_h)
                c_x = int(x1n * ori_w)
                c_y = int(y1n * ori_h)
                x1 = max(0, int(c_x - w / 2))
                y1 = max(0, int(c_y - h / 2))
                x2 = int(c_x + w / 2)
                y2 = int(c_y + h / 2)
                preds.append({
                    "bbox": [x1, y1, w, h],
                    "track_id": int(ids[idx].item()),
                    "category_id": 1,
                    "image_id": image_id,
                    "video_id": video_id,
                    "score": float(det[-1].item()),
                    "segmentations": rles[idx],
                })
        target = out_dir / f"{str(image_id).zfill(10)}.json"
        fd, tmp = tempfile.mkstemp(prefix="p4i_", suffix=".json", dir=out_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(preds, f, separators=(",", ":"))
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return log_rows


def frames_from_pre_assoc(export_root, video_id):
    frames = []
    seen = set()
    det_file = export_root / "pre_assoc_detections" / f"{video_id}.jsonl"
    for line in det_file.read_text().splitlines():
        r = json.loads(line)
        fo = r["frame_order"]
        if fo in seen:
            continue
        seen.add(fo)
        frames.append({"frame_order": fo, "image_id": r["image_id"]})
    frames.sort(key=lambda x: x["frame_order"])
    return frames
