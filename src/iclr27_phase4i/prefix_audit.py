"""Trajectory-prefix stability and prefix positive-evidence audits.

Uses GT-linked diagnostic tracks (IoU >= 0.5 between pre-association
detections and TAO GT annotations).  For each GT track we form causal
prefixes of length 1/2/4/8/16/32+ frames and report:
  - adapted prefix embedding (frozen M2 aggregator, causal mean);
  - known/unknown routing accuracy and known-class accuracy;
  - cosine to the full diagnostic track mean (offline reference, NOT
    FRAME-ONLINE);
  - semantic stability across adjacent prefixes;
and prefix-pair cosine statistics:
  - same-track prefixes, same-class different-track, different-class.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
TAO_JSON = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
KNOWN_IDS = ROOT / "data" / "trackocd_v1" / "pure" / "splits" / "supported_known_ids.json"
PRE = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "pre_assoc_detections"
FEAT_ROOT = ROOT / "outputs" / "iclr27_phase4i" / "audit" / "detection_features"

from src.orbit_mdc.evaluate_mdc import load_mdc_model
from src.orbit.protocol import load_frame_features, load_train_labels
from src.orbit.evaluate import embed_track


def _norm(x):
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def build_known(model, device):
    labels = load_train_labels()
    feats = load_frame_features("train_known_mean")
    by_class = defaultdict(list)
    for sid, c in labels.items():
        if sid in feats:
            by_class[int(c)].append(sid)
    protos, radii = {}, {}
    for c, ids in by_class.items():
        zs = []
        for sid in ids:
            z, _ = embed_track(model, feats[sid], device)
            zs.append(z)
        Z = np.stack(zs)
        p = _norm(Z.mean(axis=0).astype(np.float32))
        protos[int(c)] = p
        cos = Z @ p
        radii[int(c)] = float(np.percentile(1.0 - cos, 50).clip(min=0.02))
    return protos, radii


def embed_prefix(model, feats, device):
    """Adapted embedding of a causal prefix (mean over frames)."""
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        z = model.aggregate(x, mask)["z"][0].cpu().numpy().astype(np.float32)
    return _norm(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-pairs-csv", required=True, type=Path)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, _ = load_mdc_model("runs/orbit_mdc/mdc_m2/model.pth", device)
    model.eval()
    protos, radii = build_known(model, device)
    P = np.stack(list(protos.values())).astype(np.float32)
    ids = list(protos)
    known = set(json.loads(KNOWN_IDS.read_text()))
    tao = json.loads(TAO_JSON.read_text())
    gt_by_img = defaultdict(list)
    for ann in tao["annotations"]:
        b = ann["bbox"]
        gt_by_img[ann["image_id"]].append({
            "bbox": [b[0], b[1], b[0] + b[2], b[1] + b[3]],
            "track_id": int(ann["track_id"]),
            "category_id": int(ann["category_id"]),
            "role": "known" if int(ann["category_id"]) in known else "novel",
        })

    # per GT track, detections with their DINO features (causal order)
    gt_tracks = defaultdict(list)
    for vid in sorted(int(p.stem) for p in PRE.glob("*.jsonl")):
        feats_all = np.load(FEAT_ROOT / str(vid) / "feats.npz")
        k = 0
        for line in (PRE / f"{vid}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            gts = gt_by_img.get(int(r["image_id"]), [])
            best, bi = None, 0.5
            for g in gts:
                v = iou(r["bbox_xyxy_original"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            if best is not None and k < feats_all["feats"].shape[0]:
                gt_tracks[(int(r["video_id"]), best["track_id"])].append({
                    "frame_order": int(r["frame_order"]),
                    "feat": feats_all["feats"][k],
                    "score": float(r["score"]),
                    "category": best["category_id"],
                    "role": best["role"],
                })
            k += 1

    # prefix analysis
    buckets = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 10 ** 6)]
    rows = []
    for (vid, tid), dets in gt_tracks.items():
        dets = sorted(dets, key=lambda d: d["frame_order"])
        feats = np.stack([d["feat"] for d in dets])
        full_z = embed_prefix(model, feats, device)
        role = dets[0]["role"]
        for (lo, hi) in buckets:
            n = len(dets)
            if n < lo:
                continue
            m = min(n, hi)
            pref = feats[:m]
            z_p1 = embed_prefix(model, pref, device)
            z_p0 = embed_prefix(model, pref[-1:], device)
            w = np.asarray([d["score"] for d in dets[:m]], dtype=np.float32)
            w = w / (w.sum() + 1e-9)
            zs = [embed_prefix(model, pref[i:i + 1], device)
                  for i in range(m)]
            z_p2 = _norm(np.average(np.stack(zs), axis=0, weights=w)
                         .astype(np.float32))
            out = {"video_id": vid, "gt_track_id": tid, "role": role,
                   "category": dets[0]["category"], "prefix_len": m}
            for mode, z in (("P0", z_p0), ("P1", z_p1), ("P2", z_p2)):
                ks = P @ z
                pred_role = "known" if float(ks.max()) >= 0.5 else "novel"
                pred_class = int(ids[int(np.argmax(ks))])
                out[f"routing_correct_{mode}"] = int(pred_role == role)
                out[f"known_class_correct_{mode}"] = (
                    int(pred_class == dets[0]["category"])
                    if role == "known" else "")
                out[f"cos_to_full_mean_{mode}"] = float(np.dot(z, full_z))
            rows.append({
                **out,
            })

    # pair statistics: same track (different prefix lengths), same class
    # different track, different class (prefix length 8 whenever available)
    pair_rows = []
    z8 = {}
    z4 = {}
    for (vid, tid), dets in gt_tracks.items():
        dets = sorted(dets, key=lambda d: d["frame_order"])
        if len(dets) >= 8:
            z = embed_prefix(model, np.stack([d["feat"] for d in dets[:8]]),
                             device)
            z8[(vid, tid, dets[0]["category"])] = z
        if len(dets) >= 4:
            z = embed_prefix(model, np.stack([d["feat"] for d in dets[:4]]),
                             device)
            z4[(vid, tid, dets[0]["category"])] = z
    keys = list(z8)
    same_track, same_class, diff_class = [], [], []
    for k, v in z4.items():
        if k in z8:
            same_track.append(float(np.dot(v, z8[k])))
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = keys[i]
            b = keys[j]
            cos = float(np.dot(z8[a], z8[b]))
            if a[2] == b[2]:
                same_class.append(cos)
            else:
                diff_class.append(cos)

    def stat(name, vals):
        return {"group": name, "n": len(vals),
                "mean": float(np.mean(vals)) if vals else "",
                "median": float(np.median(vals)) if vals else "",
                "p90": float(np.percentile(vals, 90)) if vals else ""}

    pair_rows = [stat("same_track_prefix4_vs_8", same_track),
                 stat("same_class_diff_track_prefix8", same_class),
                 stat("different_class_prefix8", diff_class)]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(args.out_pairs_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)
    print(json.dumps(pair_rows, indent=1))
    print("prefix rows", len(rows))


if __name__ == "__main__":
    main()
