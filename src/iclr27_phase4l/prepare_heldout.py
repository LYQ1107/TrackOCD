"""Phase 4L held-out subset selection and COCO export JSON.

Universe: all TAO val videos in
third_party/SimOWT/datasets/tao/annotations/val_split/all.json whose
frame files exist under data/raw/tao/frames.  Dev videos used by Phases
4I-4K are excluded.  Selection is deterministic (seed 20260808),
stratified by source domain, and never based on method performance.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ALL_JSON = ROOT / "third_party" / "SimOWT" / "datasets" / "tao" / \
    "annotations" / "val_split" / "all.json"
DEV_CSV = ROOT / "outputs" / "iclr27_phase3a" / "smoke" / \
    "selected_20_videos.csv"
OUT_DIR = ROOT / "outputs" / "iclr27_phase4l" / "heldout"
SEED = 20260808


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    all_data = json.loads(ALL_JSON.read_text())
    dev_ids = set()
    with open(DEV_CSV) as f:
        for r in csv.DictReader(f):
            dev_ids.add(int(r["video_id"]))
    videos = {v["id"]: v for v in all_data["videos"]}
    images = all_data["images"]
    img_by_video = defaultdict(list)
    for im in images:
        img_by_video[im["video_id"]].append(im)

    universe = []
    missing_frames = []
    for vid, v in videos.items():
        if vid in dev_ids:
            continue
        ims = img_by_video.get(vid, [])
        if not ims:
            continue
        # frame availability check on a few sample frames
        ok = 0
        for im in ims[:: max(1, len(ims) // 5)][:5]:
            p = ROOT / "data" / "raw" / "tao" / "frames" / im["file_name"]
            if p.exists():
                ok += 1
        if ok == 0:
            missing_frames.append(vid)
            continue
        name = v["name"]
        domain = name.split("/")[1] if name.count("/") >= 2 else "other"
        universe.append({
            "video_id": vid, "video_name": name, "source_domain": domain,
            "frame_count": len(ims),
        })
    print("universe", len(universe), "dev_excluded", len(dev_ids),
          "no_frames", len(missing_frames))

    by_domain = defaultdict(list)
    for u in universe:
        by_domain[u["source_domain"]].append(u)
    rng = random.Random(SEED)
    selected = []
    per_domain = {d: len(v) for d, v in by_domain.items()}
    total = len(universe)
    quota = {d: max(1, round(args.n * n / total)) for d, n in
             per_domain.items()}
    # deterministic quota fill (seed used only for tie-breaking order)
    for d in sorted(by_domain):
        pool = sorted(by_domain[d], key=lambda u: u["video_id"])
        rng.shuffle(pool)
        selected.extend(pool[: quota[d]])
    if len(selected) > args.n:
        rng.shuffle(selected)
        selected = selected[: args.n]
    selected.sort(key=lambda u: u["video_id"])
    sel_ids = {u["video_id"] for u in selected}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "selected_heldout_videos.csv", "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "video_name", "source_domain", "frame_count",
            "selection_seed", "selection_rule"])
        w.writeheader()
        for u in selected:
            w.writerow({
                "video_id": u["video_id"],
                "video_name": u["video_name"],
                "source_domain": u["source_domain"],
                "frame_count": u["frame_count"],
                "selection_seed": SEED,
                "selection_rule": "deterministic domain-stratified seed "
                "%d, dev-excluded" % SEED,
            })
    (OUT_DIR / "universe.json").write_text(json.dumps({
        "seed": SEED, "universe_size": len(universe),
        "dev_excluded": sorted(dev_ids),
        "videos_without_frames": sorted(missing_frames),
        "domain_counts": dict(per_domain),
    }, indent=1))

    sel_images = [im for im in images if im["video_id"] in sel_ids]
    sel_image_ids = {im["id"] for im in sel_images}
    sel_anns = [a for a in all_data["annotations"]
                if a["image_id"] in sel_image_ids]
    subset = {
        "info": all_data.get("info", {}),
        "licenses": all_data.get("licenses", []),
        "categories": all_data.get("categories", []),
        "videos": [videos[v] for v in sorted(sel_ids)],
        "images": sel_images,
        "annotations": sel_anns,
    }
    out_json = OUT_DIR / "validation_heldout_coco.json"
    out_json.write_text(json.dumps(subset, separators=(",", ":")))
    print("wrote", out_json)
    print("selected", len(selected), "images", len(sel_images),
          "annotations", len(sel_anns))
    print("HELDOUT_PREPARED", sorted(sel_ids))


if __name__ == "__main__":
    main()
