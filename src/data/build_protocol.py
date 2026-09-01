#!/usr/bin/env python3
"""Build the TAO-OW TrackOCD v1 protocol.

- official known / distractor / unknown splits from Open-World-Tracking maps
- public GT track stream (no unknown labels) + private labels
- Full / Repeated / Balanced evaluation subsets
- deterministic stream orders (main + 3 seeded video-block orders)
- dataset statistics
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.data.tao_io import (
    PROJECT_ROOT,
    category_sets,
    group_tracks,
    load_tao_annotations,
    atomic_write_text,
)

OUT = PROJECT_ROOT / "data" / "tao_ow_ocd_v1"
PUBLIC = OUT / "public"
PRIVATE = OUT / "private"
SPLITS = OUT / "splits"
STATS = OUT / "stats"
MANIFESTS = OUT / "manifests"


def write_jsonl(path: Path, rows):
    text = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    atomic_write_text(path, text)


def track_to_stream_row(track, img_by_id):
    anns = track["annotations"]
    frame_ids = [a["image_id"] for a in anns]
    image_paths = [img_by_id[i]["file_name"] for i in frame_ids]
    boxes_xyxy = []
    areas = []
    for a in anns:
        x, y, w, h = a["bbox"]
        boxes_xyxy.append([x, y, x + w, y + h])
        areas.append(w * h)
    return {
        "sample_id": track["sample_id"],
        "video_id": track["video_id"],
        "track_id": track["track_id"],
        "frame_ids": frame_ids,
        "image_paths": image_paths,
        "boxes_xyxy": boxes_xyxy,
        "areas": areas,
        "stream_order": 0,
    }


def stream_order_main(tracks):
    """video_id asc -> track first-frame asc -> track_id asc."""
    return sorted(
        tracks.values(),
        key=lambda t: (t["video_id"], t["first_frame"], t["track_id"]),
    )


def stream_order_seeded(tracks, seed):
    rng = random.Random(seed)
    videos = sorted({t["video_id"] for t in tracks.values()})
    rng.shuffle(videos)
    vid_rank = {v: i for i, v in enumerate(videos)}
    return sorted(
        tracks.values(),
        key=lambda t: (vid_rank[t["video_id"]], t["first_frame"], t["track_id"]),
    )


def build_stats(train_tracks, val_tracks, known, distractor):
    stats = {"known_ids": sorted(known), "distractor_ids": sorted(distractor)}
    for split_name, tracks in (("train", train_tracks), ("val", val_tracks)):
        cats = set(t["category_id"] for t in tracks.values())
        unknown = cats - known - distractor
        stats[f"unknown_ids_{split_name}"] = sorted(unknown)
        stats[f"num_categories_{split_name}"] = len(cats)
        stats[f"num_known_categories_{split_name}"] = len(cats & known)
        stats[f"num_unknown_categories_{split_name}"] = len(unknown)
        stats[f"num_distractor_categories_{split_name}"] = len(cats & distractor)

    all_val = list(val_tracks.values())
    cat_meta = {c["id"]: c for c in load_tao_annotations("val")["categories"]}
    cat_tracks = defaultdict(int)
    cat_videos = defaultdict(set)
    cat_boxes = Counter()
    for t in all_val:
        cat_tracks[t["category_id"]] += 1
        cat_videos[t["category_id"]].add(t["video_id"])
        cat_boxes[t["category_id"]] += len(t["annotations"])

    rows = []
    for cat_id in sorted(cat_tracks):
        is_k = cat_id in known
        is_d = cat_id in distractor
        name = cat_meta.get(cat_id, {}).get("name", "?")
        rows.append(
            {
                "category_id": cat_id,
                "name": name,
                "group": "known" if is_k else ("distractor" if is_d else "unknown"),
                "num_tracks": cat_tracks[cat_id],
                "num_videos": len(cat_videos[cat_id]),
                "num_boxes": cat_boxes[cat_id],
            }
        )
    with open(STATS / "category_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    track_rows = []
    for t in all_val:
        track_rows.append(
            {
                "sample_id": t["sample_id"],
                "video_id": t["video_id"],
                "track_id": t["track_id"],
                "category_id": t["category_id"],
                "group": "known" if t["is_known"] else ("distractor" if t["is_distractor"] else "unknown"),
                "num_frames": t["num_frames"],
            }
        )
    with open(STATS / "track_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        w.writeheader()
        w.writerows(track_rows)

    nd = [t for t in all_val if not t["is_distractor"]]
    nk = sum(t["is_known"] for t in nd)
    nu = len(nd) - nk
    stats["val_num_tracks_total"] = len(all_val)
    stats["val_num_tracks_non_distractor"] = len(nd)
    stats["val_num_tracks_known"] = nk
    stats["val_num_tracks_unknown"] = nu
    stats["val_num_boxes_non_distractor"] = sum(len(t["annotations"]) for t in nd)
    stats["val_num_boxes_known"] = sum(len(t["annotations"]) for t in nd if t["is_known"])
    stats["val_num_boxes_unknown"] = sum(len(t["annotations"]) for t in nd if not t["is_known"])

    unknown_tracks = [t for t in nd if not t["is_known"]]
    unk_cat_counts = Counter(t["category_id"] for t in unknown_tracks)
    unk_cat_videos = defaultdict(set)
    for t in unknown_tracks:
        unk_cat_videos[t["category_id"]].add(t["video_id"])
    stats["unknown_categories_with_1_track"] = sum(1 for c in unk_cat_counts if unk_cat_counts[c] == 1)
    stats["unknown_categories_with_1_video"] = sum(1 for c in unk_cat_videos if len(unk_cat_videos[c]) == 1)
    stats["unknown_track_count_ratio"] = nu / len(nd) if nd else 0.0
    stats["unknown_box_count_ratio"] = stats["val_num_boxes_unknown"] / stats["val_num_boxes_non_distractor"]
    return stats


def build_subsets(val_tracks, known, distractor):
    nd = {k: t for k, t in val_tracks.items() if not t["is_distractor"]}
    full_ids = sorted(t["sample_id"] for t in nd.values())

    unknown = [t for t in nd.values() if not t["is_known"]]
    cat_tracks = defaultdict(list)
    cat_videos = defaultdict(set)
    for t in unknown:
        cat_tracks[t["category_id"]].append(t)
        cat_videos[t["category_id"]].add(t["video_id"])

    repeated_cats = [c for c, ts in cat_tracks.items() if len(ts) >= 2]
    repeated_ids = sorted(t["sample_id"] for c in repeated_cats for t in cat_tracks[c])

    bal_cats = [c for c, ts in cat_tracks.items() if len(ts) >= 3 and len(cat_videos[c]) >= 2]
    bal_cats.sort()
    rng = random.Random(20260805)
    rng.shuffle(bal_cats)
    n_bal = min(78, len(bal_cats))
    chosen = sorted(bal_cats[:n_bal])
    balanced_ids = sorted(t["sample_id"] for c in chosen for t in cat_tracks[c])
    subset_stats = {
        "full": {
            "num_tracks": len(full_ids),
            "num_unknown_categories": len(set(t["category_id"] for t in unknown)),
        },
        "repeated": {
            "num_tracks": len(repeated_ids),
            "num_unknown_categories": len(repeated_cats),
        },
        "balanced": {
            "num_tracks": len(balanced_ids),
            "num_unknown_categories": len(chosen),
            "num_eligible_categories": len(bal_cats),
            "n_requested": 78,
            "n_selected": n_bal,
        },
    }
    for name, ids in (
        ("full", full_ids),
        ("repeated", repeated_ids),
        ("balanced", balanced_ids),
    ):
        atomic_write_text(MANIFESTS / f"{name}_track_ids.json", json.dumps(ids, indent=1))
    return subset_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-frames", action="store_true", help="one-time frame integrity scan")
    args = ap.parse_args()

    known, distractor = category_sets()
    train_data = load_tao_annotations("train")
    val_data = load_tao_annotations("val")
    train_tracks = group_tracks(train_data, exclude_distractor=True)
    val_tracks = group_tracks(val_data, exclude_distractor=True)

    stats = build_stats(train_tracks, val_tracks, known, distractor)
    subset_stats = build_subsets(val_tracks, known, distractor)
    stats["subsets"] = subset_stats
    atomic_write_text(STATS / "dataset_stats.json", json.dumps(stats, indent=1, sort_keys=True))

    atomic_write_text(SPLITS / "known_ids.json", json.dumps(sorted(known), indent=1))
    atomic_write_text(SPLITS / "distractor_ids.json", json.dumps(sorted(distractor), indent=1))
    atomic_write_text(SPLITS / "unknown_ids_train.json", json.dumps(stats["unknown_ids_train"], indent=1))
    atomic_write_text(SPLITS / "unknown_ids_val.json", json.dumps(stats["unknown_ids_val"], indent=1))

    img_by_id = {im["id"]: im for im in val_data["images"]}
    train_img_by_id = {im["id"]: im for im in train_data["images"]}

    main_order = stream_order_main(val_tracks)
    for i, t in enumerate(main_order):
        t["stream_order"] = i
    write_jsonl(PUBLIC / "val_gt_track_stream.jsonl", [track_to_stream_row(t, img_by_id) for t in main_order])
    train_known_rows = [
        t
        for t in sorted(train_tracks.values(), key=lambda x: (x["video_id"], x["first_frame"], x["track_id"]))
        if t["is_known"]
    ]
    write_jsonl(
        PUBLIC / "train_known_tracks.jsonl",
        [
            {
                "sample_id": t["sample_id"],
                "video_id": t["video_id"],
                "track_id": t["track_id"],
                "category_id": t["category_id"],
                "image_paths": [train_img_by_id[a["image_id"]]["file_name"] for a in t["annotations"]],
                "boxes_xyxy": [
                    [a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                    for a in t["annotations"]
                ],
            }
            for t in train_known_rows
        ],
    )
    write_jsonl(
        PRIVATE / "val_gt_track_labels.jsonl",
        [
            {
                "sample_id": t["sample_id"],
                "ground_truth_category_id": t["category_id"],
                "is_known": t["is_known"],
                "is_distractor": t["is_distractor"],
            }
            for t in sorted(val_tracks.values(), key=lambda x: (x["video_id"], x["first_frame"], x["track_id"]))
        ],
    )

    for seed in (1027, 1028, 1029):
        order = stream_order_seeded(val_tracks, seed)
        for i, t in enumerate(order):
            t["stream_order"] = i
        write_jsonl(
            PUBLIC / f"val_gt_track_stream_seed{seed}.jsonl",
            [track_to_stream_row(t, img_by_id) for t in order],
        )

    scan_report = {"checked_images": 0, "missing_frames": [], "broken_links": 0, "invalid_boxes": 0}
    if args.scan_frames:
        from src.data.tao_io import FRAMES_ROOT, resolve_frame_path

        for im in val_data["images"]:
            p = resolve_frame_path(im)
            scan_report["checked_images"] += 1
            if not p.exists():
                scan_report["missing_frames"].append(im["file_name"])
        scan_report["broken_links"] = sum(1 for _ in FRAMES_ROOT.glob("**/*") if _.is_symlink() and not _.exists())
        for a in val_data["annotations"]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                scan_report["invalid_boxes"] += 1
        atomic_write_text(OUT / "stats" / "frame_scan.json", json.dumps(scan_report, indent=1))
        print(json.dumps(scan_report, indent=1))

    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
