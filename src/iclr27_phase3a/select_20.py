#!/usr/bin/env python3
"""Deterministic 20-video selection (domain x duration strata, SHA256) and
TAO subset json builder. No performance/Gt information is used."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
OUT = PROJECT_ROOT / "outputs" / "iclr27_phase3a"
DOCS = PROJECT_ROOT / "docs" / "iclr27_phase3a"
SALT = "trackocd_phase3a_v1"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    gt = json.load(open(PROJECT_ROOT / "data/raw/tao/annotations/validation.json"))
    vid2name = {v["id"]: v["name"] for v in gt["videos"]}
    vid2ds = {v["id"]: (v.get("metadata") or {}).get("dataset", "unknown")
              for v in gt["videos"]}
    frame_count = defaultdict(int)
    for im in gt["images"]:
        frame_count[im["video_id"]] += 1
    # strata: domain x duration bin (short<=20, mid<=60, long>60)
    def dur_bin(n):
        return "short" if n <= 20 else ("mid" if n <= 60 else "long")
    strata = defaultdict(list)
    for vid in gt["videos"]:
        ds = vid2ds[vid["id"]]
        n = frame_count[vid["id"]]
        h = hashlib.sha256(f"{vid['id']}{SALT}".encode()).hexdigest()
        strata[(ds, dur_bin(n))].append((h, vid["id"], n))
    picked = []
    for key, items in sorted(strata.items()):
        if not items:
            continue
        items.sort()
        picked.append(items[0])
    # fill to 20 by global hash
    all_items = []
    for items in strata.values():
        all_items.extend(items)
    all_items.sort()
    seen = {v for _, v, _ in picked}
    for it in all_items:
        if len(picked) >= 20:
            break
        if it[1] not in seen:
            picked.append(it)
            seen.add(it[1])
    picked = sorted(picked)[:20]
    rows = [{
        "video_id": vid, "video_name": vid2name[vid],
        "source_domain": vid2ds[vid], "frame_count": frame_count[vid],
        "duration_bin": dur_bin(frame_count[vid]),
        "selection_hash": h, "selection_reason": "deterministic SHA256 strata minimum",
    } for h, vid, _ in picked]
    write_csv(OUT / "smoke/selected_20_videos.csv", rows)
    selected_ids = {r["video_id"] for r in rows}
    # subset json (copy of raw validation restricted to 20 videos)
    sub = {
        "info": gt.get("info", {}), "licenses": gt.get("licenses", []),
        "categories": gt["categories"],
        "videos": [v for v in gt["videos"] if v["id"] in selected_ids],
        "images": [im for im in gt["images"] if im["video_id"] in selected_ids],
        "tracks": [t for t in gt["tracks"] if t["video_id"] in selected_ids],
        "annotations": [a for a in gt["annotations"] if a["video_id"] in selected_ids],
    }
    out_json = OUT / "smoke/tao_subset/validation_20.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(sub))
    (DOCS / "SMOKE_VIDEO_SELECTION.md").write_text(
        "# Smoke Video Selection\n\n20 videos selected deterministically by "
        "SHA256(video_id + trackocd_phase3a_v1) within source-domain x "
        "duration-bin strata, then global hash fill. No performance/GT "
        "information used. See `selected_20_videos.csv`.\n")
    print("selected", len(rows), "domains", sorted({r["source_domain"] for r in rows}),
          "frames", sum(r["frame_count"] for r in rows))
    for r in rows:
        print(r["video_id"], r["source_domain"], r["duration_bin"], r["frame_count"])


if __name__ == "__main__":
    main()
