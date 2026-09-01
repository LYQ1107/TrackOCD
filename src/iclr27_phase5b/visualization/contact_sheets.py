"""Build contact sheets for a stratified sample of unaligned tracks.

Each sheet shows first / middle / last crop and full-frame bbox context.
This is a diagnostic aid for MANUAL_VISUAL_AUDIT_REQUIRED; the images are not
used as ground truth.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.iclr27_phase4s.protocol import (
    Q1_DEV,
    TAO_VAL_ANN,
    group_tracks,
    load_proposals,
)

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(Q1_DEV))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--out", default="outputs/iclr27_phase5b/visual_audit")
    args = ap.parse_args()

    val = json.loads(TAO_VAL_ANN.read_text())
    file_of_img = {int(im["id"]): im["file_name"] for im in val["images"]}
    rows = load_proposals(Path(args.csv))
    tracks = group_tracks(rows)

    # forensic causes
    cause = {}
    with open(ROOT / "outputs/iclr27_phase5b/audit/geometry/track_forensic_table.csv") as f:
        for r in csv.DictReader(f):
            if r["aligned"] == "0":
                cause[(int(r["video_id"]), int(r["track_id"]))] = r["forensic_cause"]
    unaligned = [k for k in tracks if k not in cause]  # shouldn't happen
    unaligned = [k for k in tracks if cause.get(k)]

    def bucket_len(n):
        if n <= 1: return "len1"
        if n == 2: return "len2"
        if n == 3: return "len3"
        if n <= 10: return "len4_10"
        return "len_gt10"

    def bucket_score(tr):
        sc = np.mean([r["score"] for r in tr])
        if sc < 0.3: return "score_lt0.3"
        if sc < 0.5: return "score_0.3_0.5"
        return "score_ge0.5"

    strata = defaultdict(list)
    for k in unaligned:
        strata[(bucket_len(len(tracks[k])), bucket_score(tracks[k]),
                int(tracks[k][0]["video_id"]), cause[k])].append(k)
    # deterministic stratified sample
    rng = random.Random(args.seed)
    chosen = []
    # first pass: take up to 1 per stratum, then fill remaining by length/score
    for st, keys in sorted(strata.items()):
        chosen.append(rng.choice(keys))
    pool = [k for k in unaligned if k not in set(chosen)]
    rng.shuffle(pool)
    chosen += pool[: max(0, args.n - len(chosen))]
    chosen = chosen[: args.n]

    out = ROOT / args.out
    sheets = out / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, k in enumerate(chosen):
        tr = sorted(tracks[k], key=lambda r: r["frame_id"])
        idxs = sorted(set(range(len(tr))) & {0, len(tr) // 2, len(tr) - 1})
        panels = []
        for t in idxs:
            r = tr[t]
            fp = ROOT / "data/raw/tao/frames" / file_of_img[int(r["image_id"])]
            img = cv2.imread(str(fp))
            if img is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in json.loads(r["bbox_xyxy"])]
            h, w = img.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)
            crop = cv2.resize(crop, (180, 180), interpolation=cv2.INTER_AREA)
            full = img.copy()
            cv2.rectangle(full, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(full, f"{k[1]} s={r['score']:.2f} c={r['category_id']}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            full = cv2.resize(full, (240, 180), interpolation=cv2.INTER_AREA)
            panels.append(np.hstack([crop, full]))
        if not panels:
            continue
        sheet = np.vstack(panels) if len(panels) > 1 else panels[0]
        name = f"{i:04d}_v{k[0]}_t{k[1]}_{cause[k]}.png"
        cv2.imwrite(str(sheets / name), sheet)
        manifest.append({
            "sheet": name, "video_id": int(k[0]), "track_id": int(k[1]),
            "length": len(tr), "cause": cause[k],
            "mean_score": float(np.mean([r["score"] for r in tr])),
            "frames": [int(r["frame_id"]) for r in [tr[t] for t in idxs]],
        })
    with open(out / "sample_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print("sheets", len(manifest), "->", sheets)


if __name__ == "__main__":
    main()
