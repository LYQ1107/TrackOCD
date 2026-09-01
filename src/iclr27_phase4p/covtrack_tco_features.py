#!/usr/bin/env python3
"""Extract causal trajectory features for COVTrack proposals (dev/heldout).

All temporal features for proposal at frame t use only the same COVTrack
physical track_id at frames < t. DINO crop embeddings are cached per split.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
sys.path.insert(0, str(ROOT))

from src.data.tao_io import FRAMES_ROOT
from src.features.extract import crop_bbox, load_dinov2, make_crop_transform

OUT = ROOT / "outputs" / "iclr27_phase4p" / "covtrack_tco"


def build_known_prototypes():
    cache = ROOT / "data" / "caches" / "features" / "dinov2" / "train_known_mean"
    feats = {}
    for p in cache.glob("*.json"):
        r = json.loads(p.read_text())
        feats[r["sample_id"]] = np.asarray(r["mean_embedding"], dtype=np.float32)
    labels = {}
    with open(ROOT / "data" / "tao_ow_ocd_v1" / "public" / "train_known_tracks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["sample_id"] in feats:
                labels[r["sample_id"]] = r["category_id"]
    sums = defaultdict(lambda: np.zeros(768, dtype=np.float64))
    counts = defaultdict(int)
    for sid, c in labels.items():
        sums[c] += feats[sid]
        counts[c] += 1
    protos = {}
    for c, s in sums.items():
        v = s / counts[c]
        protos[c] = v / (np.linalg.norm(v) + 1e-12)
    return protos


def load_population(mode):
    rows = list(csv.DictReader(open(OUT / f"proposal_population_{mode}.csv")))
    for r in rows:
        r["video_id"] = int(r["video_id"])
        r["frame_id"] = int(r["frame_id"])
        r["image_id"] = int(r["image_id"])
        r["proposal_local_id"] = int(r["proposal_local_id"])
        r["track_id"] = int(r["track_id"])
        r["score"] = float(r["score"])
        r["bbox_xyxy"] = json.loads(r["bbox_xyxy"])
        r["bbox_area"] = float(r["bbox_area"])
        r["bbox_aspect"] = float(r["bbox_aspect"])
    return rows


def extract_dino(rows, mode):
    cache = OUT / f"dino_embeds_{mode}.npz"
    if cache.exists():
        z = np.load(cache)
        embeds = z["embeds"]
        raw_norm = z["raw_norm"]
        print(f"[cache] {mode} embeds {embeds.shape}", flush=True)
        return embeds, raw_norm

    model, size = load_dinov2()
    transform = make_crop_transform(size)
    image_map = {}
    val = json.loads((ROOT / "data" / "raw" / "tao" / "annotations" / "validation.json").read_text())
    for im in val["images"]:
        image_map[im["id"]] = im["file_name"]

    embeds = np.zeros((len(rows), 768), dtype=np.float32)
    raw_norm = np.zeros(len(rows), dtype=np.float32)
    ok = 0
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        tensors = []
        valid = []
        for i, r in enumerate(batch):
            p = FRAMES_ROOT / image_map[r["image_id"]]
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            crop = crop_bbox(img, r["bbox_xyxy"])
            if min(crop.size) < 4:
                continue
            tensors.append(transform(crop).unsqueeze(0).cuda())
            valid.append((i, r))
        if not valid:
            continue
        t = torch.cat(tensors, dim=0)
        with torch.no_grad():
            raw = model(t)  # not L2 normalized
        raw = raw.float().cpu().numpy()
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        raw_norm[start + np.array([v[0] for v in valid])] = norms[:, 0]
        embeds[start + np.array([v[0] for v in valid])] = raw / (norms + 1e-12)
        ok += len(valid)
        if (start // 128) % 20 == 0:
            print(f"[dino] {mode} {start}/{len(rows)} ok={ok}", flush=True)
    np.savez(cache, embeds=embeds, raw_norm=raw_norm)
    print(f"[dino] {mode} done ok={ok}/{len(rows)} -> {cache}", flush=True)
    return embeds, raw_norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    args = ap.parse_args()

    rows = load_population(args.mode)
    embeds, raw_norm = extract_dino(rows, args.mode)
    protos = build_known_prototypes()
    proto_list = list(protos.values())
    proto_arr = np.stack(proto_list)
    sims = embeds @ proto_arr.T
    top2 = np.partition(sims, -2, axis=1)[:, -2:]
    best_known = sims.max(axis=1)
    second_known = top2[:, 0]
    known_margin = best_known - second_known

    # group by track for causal history
    by_track = defaultdict(list)
    for i, r in enumerate(rows):
        by_track[(r["video_id"], r["track_id"])].append(i)
    for k in by_track:
        by_track[k].sort(key=lambda i: (rows[i]["frame_id"], rows[i]["proposal_local_id"]))

    out_rows = []
    for i, r in enumerate(rows):
        key = (r["video_id"], r["track_id"])
        hist = [j for j in by_track[key] if rows[j]["frame_id"] < r["frame_id"]]
        prior_hits = len(hist)
        prior_age = 0
        if hist:
            prior_age = r["frame_id"] - rows[hist[-1]]["frame_id"]
            # count unique prior frames
            prior_hits = len({rows[j]["frame_id"] for j in hist})
        recent = [j for j in hist if r["frame_id"] - rows[j]["frame_id"] <= 5]
        recent_hit_ratio = len({rows[j]["frame_id"] for j in recent}) / 5.0
        # consecutive hits ending at last prior frame
        consec = 0
        if hist:
            prev_frames = sorted({rows[j]["frame_id"] for j in hist})
            expect = prev_frames[-1]
            while expect in set(prev_frames):
                consec += 1
                expect -= 1
        # recent miss = frames since last hit (0 if previous frame)
        recent_miss = r["frame_id"] - rows[hist[-1]]["frame_id"] - 1 if hist else -1

        # motion from consecutive prior frames
        centers = np.asarray([
            [(rows[j]["bbox_xyxy"][0] + rows[j]["bbox_xyxy"][2]) / 2,
             (rows[j]["bbox_xyxy"][1] + rows[j]["bbox_xyxy"][3]) / 2]
            for j in hist
        ])
        areas = np.asarray([rows[j]["bbox_area"] for j in hist], dtype=np.float64)
        scores = np.asarray([rows[j]["score"] for j in hist], dtype=np.float64)
        disp = np.diff(centers, axis=0) if len(centers) >= 2 else np.zeros((0, 2))
        disp_norm = np.linalg.norm(disp, axis=1) if len(disp) else np.zeros(0)
        log_area = np.log(areas + 1e-6)
        scale_delta = np.diff(log_area) if len(log_area) >= 2 else np.zeros(0)
        prior_emb = embeds[hist] if hist else np.zeros((0, 768), dtype=np.float32)
        if len(prior_emb):
            app_sim = embeds[i] @ prior_emb.T
        else:
            app_sim = np.zeros(0)

        prior_best_known = best_known[hist] if hist else np.zeros(0)
        prior_margin = known_margin[hist] if hist else np.zeros(0)

        def stat(arr, fn):
            return float(fn(arr)) if len(arr) else 0.0

        out_rows.append(
            {
                "video_id": r["video_id"],
                "frame_id": r["frame_id"],
                "image_id": r["image_id"],
                "proposal_local_id": r["proposal_local_id"],
                "track_id": r["track_id"],
                "gt_role": r["gt_role"],
                "score": r["score"],
                "bbox_area_log": math.log(r["bbox_area"] + 1e-6),
                "bbox_aspect_log": math.log(r["bbox_aspect"] + 1e-6),
                "dino_norm": float(raw_norm[i]),
                "best_known": float(best_known[i]),
                "known_margin": float(known_margin[i]),
                "prior_age": float(prior_age),
                "prior_hits": float(prior_hits),
                "recent_hit_ratio": float(recent_hit_ratio),
                "consecutive_hits": float(consec),
                "recent_miss": float(recent_miss),
                "prior_score_mean": stat(scores, np.mean),
                "prior_score_std": stat(scores, np.std),
                "prior_area_mean": stat(areas, np.mean),
                "prior_area_std": stat(areas, np.std),
                "disp_mean": stat(disp_norm, np.mean),
                "disp_std": stat(disp_norm, np.std),
                "scale_delta_mean": stat(scale_delta, np.mean),
                "scale_delta_std": stat(scale_delta, np.std),
                "app_sim_mean": stat(app_sim, np.mean),
                "app_sim_std": stat(app_sim, np.std),
                "prior_best_known_mean": stat(prior_best_known, np.mean),
                "prior_best_known_std": stat(prior_best_known, np.std),
                "prior_margin_mean": stat(prior_margin, np.mean),
                "prior_margin_std": stat(prior_margin, np.std),
            }
        )

    out_csv = OUT / f"causal_features_{args.mode}.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"features {args.mode}: {len(out_rows)} rows -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
