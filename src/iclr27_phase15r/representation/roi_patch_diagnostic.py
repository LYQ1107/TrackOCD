"""Frozen DINOv2 ROI-patch diagnostic for the matched DSCT TRAIN subset.

This is a bounded Phase-15R/R-C diagnostic only.  It does not alter the
registered raw-CLS audit, train an adapter, or read DEV+/Q1 labels.  It pools
the DINOv2 patch tokens whose centers lie inside the proposal (with the same
fixed 10% context crop) and evaluates category-disjoint public TRAIN
correspondence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
FRAMES = ROOT / "data/iclr27_phase14c/sources/tao_train_frames"


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def crop_box(img: Image.Image, box: list[float], context: float = 0.10):
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nw, nh = bw * (1 + 2 * context), bh * (1 + 2 * context)
    xa, ya = max(0.0, cx - nw / 2), max(0.0, cy - nh / 2)
    xb, yb = min(float(w), cx + nw / 2), min(float(h), cy + nh / 2)
    if xb - xa < 2 or yb - ya < 2:
        xa, ya, xb, yb = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
    return img.crop((int(xa), int(ya), int(xb), int(yb))), (xa, ya, xb, yb)


def track_metrics(rows, feats, known_ids):
    from sklearn.metrics import average_precision_score, roc_auc_score
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("gt_role") not in ("known", "supported_known"):
            continue
        c = int(r.get("gt_category_id", -1))
        if c not in known_ids:
            continue
        by_track[(int(r["video_id"]), int(r["track_id"]))].append(i)
    tracks = []
    for key, ids in by_track.items():
        c = int(rows[ids[0]]["gt_category_id"])
        tracks.append((key, l2(feats[ids].mean(axis=0)), c))
    pairs, labels, r1 = [], [], []
    for i, (ki, vi, ci) in enumerate(tracks):
        cand = [j for j, (kj, _, _) in enumerate(tracks) if j != i and kj[0] != ki[0]]
        pos = {j for j in cand if tracks[j][2] == ci}
        if not pos:
            continue
        ranked = sorted(cand, key=lambda j: (-float(vi @ tracks[j][1]), j))
        r1.append(float(ranked[0] in pos))
        for j in cand:
            pairs.append(float(vi @ tracks[j][1])); labels.append(int(tracks[j][2] == ci))
    s, y = np.asarray(pairs, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    valid = len(np.unique(y)) > 1
    return {
        "tracks": len(tracks), "videos": len({k[0] for k, _, _ in tracks}),
        "categories": len({c for _, _, c in tracks}), "cross_video_pairs": len(s),
        "positive_pairs": int(y.sum()), "r1": float(np.mean(r1)) if r1 else None,
        "roc_auc": float(roc_auc_score(y, s)) if valid else None,
        "pr_auc": float(average_precision_score(y, s)) if valid else None,
        "positive_negative_gap": float(s[y == 1].mean() - s[y == 0].mean()) if valid else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposals", default="outputs/iclr27_phase15r/dsct_subset/proposals.csv")
    ap.add_argument("--annotation", default="data/iclr27_phase15r/sources/validation_train_subset.json")
    ap.add_argument("--out", default="outputs/iclr27_phase15r/dsct_subset/proposal_dinov2_roi_patch.npz")
    ap.add_argument("--summary", default="outputs/iclr27_phase15r/eval/proposal_roi_patch_diagnostic.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    rows = list(csv.DictReader((ROOT / args.proposals).open()))
    ann = json.loads((ROOT / args.annotation).read_text())
    images = {int(x["id"]): x for x in ann["images"]}
    known = {int(x) for x in json.loads((ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json").read_text())}
    tf = transforms.Compose([
        transforms.Resize((518, 518), interpolation=Image.BILINEAR), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(args.device)
    out_feats = np.zeros((len(rows), 768), dtype=np.float32)
    cls_feats = np.zeros((len(rows), 768), dtype=np.float32)
    tensors, masks, idxs = [], [], []

    def flush():
        nonlocal tensors, masks, idxs
        if not tensors:
            return
        batch = torch.cat(tensors, 0).to(args.device)
        with torch.no_grad():
            ff = model.forward_features(batch)
            cls = torch.nn.functional.normalize(ff["x_norm_clstoken"], dim=-1).cpu().numpy()
            patch = ff["x_norm_patchtokens"].reshape(len(idxs), 37, 37, 768)
            patch = patch.cpu().numpy()
        for j, i in enumerate(idxs):
            sel = patch[j][masks[j]]
            if sel.shape[0] == 0:
                sel = patch[j].reshape(-1, 768)
            cls_feats[i] = cls[j]
            out_feats[i] = l2(sel.mean(axis=0))
        tensors, masks, idxs = [], [], []

    for i, r in enumerate(rows):
        im = images[int(r["image_id"])]
        box = json.loads(r["bbox_xyxy"])
        path = FRAMES / im["file_name"]
        with Image.open(path) as raw:
            crop, cb = crop_box(raw.convert("RGB"), box)
        xa, ya, xb, yb = cb
        # Proposal coordinates in the resized 518x518 crop.
        sx, sy = 518.0 / max(xb - xa, 1e-6), 518.0 / max(yb - ya, 1e-6)
        rx1, ry1 = (box[0] - xa) * sx, (box[1] - ya) * sy
        rx2, ry2 = (box[2] - xa) * sx, (box[3] - ya) * sy
        centers = (np.arange(37, dtype=np.float32) + 0.5) * 14.0
        mask = ((centers[None, :] >= rx1) & (centers[None, :] <= rx2) &
                (centers[:, None] >= ry1) & (centers[:, None] <= ry2))
        tensors.append(tf(crop).unsqueeze(0)); masks.append(mask); idxs.append(i)
        if len(tensors) >= args.batch:
            flush()
    flush()
    assert np.isfinite(out_feats).all() and np.all(np.linalg.norm(out_feats, axis=1) > 0.99)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp"); np.savez_compressed(tmp, feats=out_feats, cls_feats=cls_feats)
    generated = Path(str(tmp) + ".npz") if not str(tmp).endswith(".npz") else tmp
    os.replace(generated, out)
    summary = {
        "protocol": "phase15r_r-c_diagnostic", "method": "DINOv2 ROI patch-token mean",
        "context": 0.10, "input_rows": len(rows), "feature_shape": list(out_feats.shape),
        "future_frames_used": False, "q1_label_used": False,
        "physical_id_used_as_feature": False, "gt_labels_used_for_alignment_only": True,
        "cls_track_correspondence": track_metrics(rows, cls_feats, known),
        "roi_patch_track_correspondence": track_metrics(rows, out_feats, known),
        "source_proposals": str((ROOT / args.proposals).resolve()),
        "source_frames": str(FRAMES.resolve()), "output": str(out.resolve()),
    }
    sp = ROOT / args.summary; sp.parent.mkdir(parents=True, exist_ok=True)
    st = sp.with_suffix(sp.suffix + ".tmp"); st.write_text(json.dumps(summary, indent=2, sort_keys=True)); os.replace(st, sp)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
