#!/usr/bin/env python3
"""Convert OVTR/COVTrack TETA-style tracking results to the Phase 4O
detector-only proposals CSV (filtered to dev/heldout video subsets)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")


def load_results(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "track_results" in data:
            tr = data["track_results"]
        elif "results" in data:
            tr = data["results"]
        else:
            tr = data
        # flatten per-image dict if needed
        if isinstance(tr, dict):
            out = []
            for img_id, anns in tr.items():
                for a in anns:
                    if isinstance(a, dict):
                        a = dict(a)
                        a.setdefault("image_id", int(img_id))
                    out.append(a)
            return out
        return tr
    raise ValueError(f"unrecognized result format: {type(data)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--mode", choices=["dev", "heldout"], required=True)
    args = ap.parse_args()

    gt_path = (
        ROOT / "outputs" / "iclr27_phase3a" / "smoke" / "tao_subset" / "validation_20.json"
        if args.mode == "dev"
        else ROOT / "outputs" / "iclr27_phase4n" / "audit" / "validation_heldout_tao_corrected.json"
    )
    gt = json.loads(gt_path.read_text())
    valid_img = {im["id"] for im in gt["images"]}

    rows = load_results(args.results_json)
    out = []
    skipped = 0
    for r in rows:
        img_id = r.get("image_id")
        if img_id is None:
            skipped += 1
            continue
        img_id = int(img_id)
        if img_id not in valid_img:
            continue
        bbox = r.get("bbox")
        if not bbox:
            continue
        x, y, w, h = (float(v) for v in bbox[:4])
        score = float(r.get("score", r.get("scores", 0.0)))
        out.append({
            "image_id": img_id,
            "bbox_xyxy": json.dumps([x, y, x + w, y + h]),
            "score": score,
            "track_id": r.get("track_id", r.get("instance_id", -1)),
            "category_id": r.get("category_id", r.get("label", -1)),
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "bbox_xyxy", "score", "track_id", "category_id"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"wrote {len(out)} proposals (skipped {skipped} no-image rows) -> {args.out_csv}")


if __name__ == "__main__":
    main()
