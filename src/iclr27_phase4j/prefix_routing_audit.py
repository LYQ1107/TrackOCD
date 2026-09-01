"""Prefix routing audit: real frame-online gate by prefix representation
and track age.

P0 = single-frame adapted embedding; P1 = causal running mean; P2 =
score-weighted causal mean.  For GT-linked diagnostic tracks, per age we
report routing accuracy, K2N, N2K, known-class accuracy, and semantic
switch rate using the frozen M2 gate (threshold 0.5) with known prototypes
only (no novel memory, n_novel=0).
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
from src.orbit_msr.protocol import known_stats

AGE_BUCKETS = [(1, 1, "age1"), (2, 2, "age2"), (3, 4, "age3_4"),
               (5, 8, "age5_8"), (9, 16, "age9_16"), (17, 10 ** 6, "age17plus")]


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
    x = torch.as_tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
    mask = torch.ones(1, x.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        z = model.aggregate(x, mask)["z"][0].cpu().numpy().astype(np.float32)
    return _norm(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True, type=Path)
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

    gt_tracks = defaultdict(list)
    for vid in sorted(int(p.stem) for p in PRE.glob("*.jsonl")):
        feats_all = np.load(FEAT_ROOT / str(vid) / "feats.npz")
        k = 0
        for line in (PRE / f"{vid}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            best, bi = None, 0.5
            for g in gt_by_img.get(int(r["image_id"]), []):
                v = iou(r["bbox_xyxy_original"], g["bbox"])
                if v >= bi:
                    bi, best = v, g
            if best is not None and k < feats_all["feats"].shape[0]:
                gt_tracks[(vid, best["track_id"])].append({
                    "frame_order": int(r["frame_order"]),
                    "feat": feats_all["feats"][k],
                    "score": float(r["score"]),
                    "category": best["category_id"],
                    "role": best["role"],
                })
            k += 1

    rows = []
    for (vid, tid), dets in gt_tracks.items():
        dets = sorted(dets, key=lambda d: d["frame_order"])
        feats = np.stack([d["feat"] for d in dets])
        scores = np.asarray([d["score"] for d in dets], dtype=np.float32)
        role = dets[0]["role"]
        cat = dets[0]["category"]
        prev_sem = {}
        for m in range(1, len(feats) + 1):
            pref = feats[:m]
            z_p1 = embed_prefix(model, pref, device)
            z_p0 = embed_prefix(model, pref[-1:], device)
            w = scores[:m]
            w = w / (w.sum() + 1e-9)
            zs = [embed_prefix(model, pref[i:i + 1], device)
                  for i in range(m)]
            z_p2 = _norm(np.average(np.stack(zs), axis=0, weights=w)
                         .astype(np.float32))
            for lo, hi, name in AGE_BUCKETS:
                if lo <= m <= hi:
                    bucket = name
                    break
            else:
                bucket = "age17plus"
            for mode, z in (("P0", z_p0), ("P1", z_p1), ("P2", z_p2)):
                ks = P @ z
                gs = known_stats(z, P, radii, known_ids=ids,
                                 rel=1.0, track_len=m, n_novel=0,
                                 include_anchor=False)
                with torch.no_grad():
                    logit = float(model.gate_forward(
                        torch.as_tensor([gs], dtype=torch.float32,
                                        device=device))[0])
                p_known = float(torch.sigmoid(torch.as_tensor(logit)))
                pred_role = "known" if p_known >= 0.5 else "novel"
                pred_class = int(ids[int(np.argmax(ks))])
                sem_id = f"K{pred_class}" if pred_role == "known" else "N"
                switch = int(prev_sem.get(mode) is not None and
                             prev_sem.get(mode) != sem_id)
                prev_sem[mode] = sem_id
                rows.append({
                    "video_id": vid, "gt_track_id": tid, "age": m,
                    "bucket": bucket, "mode": mode, "role": role,
                    "category": cat,
                    "routing_correct": int(pred_role == role),
                    "k2n": int(role == "known" and pred_role == "novel"),
                    "n2k": int(role == "novel" and pred_role == "known"),
                    "known_class_correct": int(
                        role == "known" and pred_role == "known"
                        and pred_class == cat),
                    "known_class_total": int(
                        role == "known" and pred_role == "known"),
                    "p_known": p_known, "semantic_switch": switch,
                })

    out_rows = []
    for mode in ("P0", "P1", "P2"):
        for bucket in [b[2] for b in AGE_BUCKETS]:
            rs = [r for r in rows if r["mode"] == mode and r["bucket"] == bucket]
            if not rs:
                continue
            known_m = [r for r in rs if r["role"] == "known"]
            novel_m = [r for r in rs if r["role"] == "novel"]
            kc = [r for r in known_m if r["known_class_total"] == 1]
            out_rows.append({
                "mode": mode, "bucket": bucket, "n": len(rs),
                "routing_accuracy": sum(r["routing_correct"] for r in rs)
                    / max(len(rs), 1),
                "k2n": sum(r["k2n"] for r in known_m) / max(len(known_m), 1),
                "n2k": sum(r["n2k"] for r in novel_m) / max(len(novel_m), 1),
                "known_class_accuracy": sum(r["known_class_correct"] for r in kc)
                    / max(len(kc), 1),
                "mean_p_known": float(np.mean([r["p_known"] for r in rs])),
                "semantic_switch_rate": sum(r["semantic_switch"] for r in rs)
                    / max(len(rs), 1),
            })
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    for r in out_rows:
        print(r)


if __name__ == "__main__":
    main()
