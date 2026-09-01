#!/usr/bin/env python3
"""Build and freeze TrackOCD-v1.0 protocols (Pure and OV-assisted).

Pure TrackOCD:
  known = official known classes with >=1 TAO-train annotation
          (train-supported known)
  novel = official unknown + official known with zero TAO-train samples

OV-assisted TrackOCD:
  known = all 78 official known classes (supported-known + zero-shot-known)
  novel = official unknown classes

All counts are computed dynamically from annotations. Public files never
contain novel category ids/names or future stream information.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LEGACY = PROJECT_ROOT / "data" / "tao_ow_ocd_v1"
RAW = PROJECT_ROOT / "data" / "raw" / "tao"
OUT_ROOT = PROJECT_ROOT / "data" / "trackocd_v1"

STREAMS = [
    "val_gt_track_stream.jsonl",
    "val_gt_track_stream_seed1027.jsonl",
    "val_gt_track_stream_seed1028.jsonl",
    "val_gt_track_stream_seed1029.jsonl",
]
SEEDS = ["main", "main_seed1027", "main_seed1028", "main_seed1029"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_private():
    rows = {}
    with open(LEGACY / "private" / "val_gt_track_labels.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["sample_id"]] = r
    return rows


def load_val_annotations():
    gt = json.load(open(RAW / "annotations" / "validation.json"))
    img2vid = {im["id"]: im["video_id"] for im in gt["images"]}
    per_track = defaultdict(lambda: {"frames": [], "boxes": [], "video": None, "category": None})
    for ann in gt["annotations"]:
        key = (ann["video_id"], ann["track_id"])
        rec = per_track[key]
        rec["video"] = ann["video_id"]
        rec["category"] = ann["category_id"]
        rec["frames"].append(ann["image_id"])
        rec["boxes"].append(ann["bbox"])
    return per_track


def main():
    known_ids = set(json.loads((LEGACY / "splits" / "known_ids.json").read_text()))
    distractor_ids = set(json.loads((LEGACY / "splits" / "distractor_ids.json").read_text()))
    unknown_ids = set(json.loads((LEGACY / "splits" / "unknown_ids_val.json").read_text()))
    assert len(known_ids) == 78 and len(distractor_ids) == 45

    train_anns = json.load(open(RAW / "annotations" / "train.json"))
    train_cats = set(a["category_id"] for a in train_anns["annotations"])
    supported_known = set(known_ids) & train_cats
    zero_shot_known = set(known_ids) - train_cats
    print(f"official known={len(known_ids)} supported={len(supported_known)} "
          f"zero-shot={len(zero_shot_known)} distractor={len(distractor_ids)} "
          f"unknown={len(unknown_ids)}", flush=True)

    private = load_private()
    val_tracks = load_val_annotations()
    _train_meta = json.load(open(RAW / "annotations" / "train.json"))
    names = {c["id"]: c["name"] for c in _train_meta["categories"]}

    protocols = {
        "pure": {
            "known": supported_known,
            "zero_shot_known": set(),
            "novel": set(unknown_ids) | zero_shot_known,
        },
        "ov_assisted": {
            "known": set(known_ids),
            "zero_shot_known": set(zero_shot_known),
            "novel": set(unknown_ids),
        },
    }

    summary = {}
    for proto, roles in protocols.items():
        root = OUT_ROOT / proto
        for sub in ("public", "private", "splits", "stats"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        # public streams: same order/features as legacy (no category info)
        for s in STREAMS:
            shutil.copyfile(LEGACY / "public" / s, root / "public" / s)
        # public train-known tracks (known categories only)
        shutil.copyfile(
            LEGACY / "public" / "train_known_tracks.jsonl",
            root / "public" / "train_known_tracks.jsonl",
        )
        # public category names: only known classes for this protocol
        known_names = {str(c): names[c] for c in sorted(roles["known"])}
        (root / "public" / "known_category_names.json").write_text(
            json.dumps(known_names, indent=1, sort_keys=True)
        )
        # splits
        (root / "splits" / "supported_known_ids.json").write_text(
            json.dumps(sorted(roles["known"])))
        (root / "splits" / "zero_shot_known_ids.json").write_text(
            json.dumps(sorted(roles["zero_shot_known"])))
        (root / "splits" / "novel_ids.json").write_text(
            json.dumps(sorted(roles["novel"])))
        (root / "splits" / "distractor_ids.json").write_text(
            json.dumps(sorted(distractor_ids)))
        # private labels with protocol_role
        label_rows = []
        for sid, r in sorted(private.items()):
            cat = r["ground_truth_category_id"]
            if cat in distractor_ids:
                role = "distractor"
            elif cat in roles["known"]:
                role = "zero_shot_known" if cat in roles["zero_shot_known"] else "supported_known"
            elif cat in roles["novel"]:
                role = "novel"
            else:
                role = "distractor"  # safety: anything unlisted is excluded
            label_rows.append(
                {
                    "sample_id": sid,
                    "ground_truth_category_id": cat,
                    "protocol_role": role,
                }
            )
        priv_path = root / "private" / "val_gt_track_labels.jsonl"
        with open(priv_path, "w") as f:
            for r in label_rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        # role counts
        role_counts = Counter(r["protocol_role"] for r in label_rows)
        role_cat_counts = Counter(
            r["protocol_role"] for r in label_rows
        )
        cat_tracks = defaultdict(lambda: {"tracks": 0, "videos": set(), "boxes": 0})
        for sid, r in private.items():
            cat = r["ground_truth_category_id"]
            rec = cat_tracks[cat]
            rec["tracks"] += 1
            vt = val_tracks.get((int(sid.split("_")[0]), int(sid.split("_")[1])))
            if vt is not None:
                rec["videos"].add(vt["video"])
                rec["boxes"] += len(vt["frames"])
        # subset masks (same rules as Architecture 1)
        full_ids = [r["sample_id"] for r in label_rows if r["protocol_role"] != "distractor"]
        novel_cat_track_count = Counter(
            r["ground_truth_category_id"] for r in label_rows if r["protocol_role"] == "novel"
        )
        novel_cat_video_count = {}
        for sid, r in private.items():
            if r["ground_truth_category_id"] not in roles["novel"]:
                continue
            vt = val_tracks.get((int(sid.split("_")[0]), int(sid.split("_")[1])))
            if vt is not None:
                novel_cat_video_count.setdefault(r["ground_truth_category_id"], set()).add(vt["video"])
        repeated_ids = [
            r["sample_id"] for r in label_rows
            if r["protocol_role"] == "novel"
            and novel_cat_track_count[r["ground_truth_category_id"]] >= 2
        ]
        balanced_ids = [
            r["sample_id"] for r in label_rows
            if r["protocol_role"] == "novel"
            and novel_cat_track_count[r["ground_truth_category_id"]] >= 3
            and len(novel_cat_video_count.get(r["ground_truth_category_id"], set())) >= 2
        ]
        (root / "splits" / "full_track_ids.json").write_text(json.dumps(full_ids))
        (root / "splits" / "repeated_track_ids.json").write_text(json.dumps(repeated_ids))
        (root / "splits" / "balanced_track_ids.json").write_text(json.dumps(balanced_ids))
        # stats
        with open(root / "stats" / "category_stats.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category_id", "name", "protocol_role", "num_tracks", "num_videos", "num_boxes"])
            for cat in sorted(cat_tracks):
                rec = cat_tracks[cat]
                role = next(
                    (r["protocol_role"] for r in label_rows if r["ground_truth_category_id"] == cat),
                    "distractor",
                )
                w.writerow([cat, names.get(cat, ""), role, rec["tracks"], len(rec["videos"]), rec["boxes"]])
        track_stats = []
        for sid, r in sorted(private.items()):
            vid, tid = sid.split("_")[:2]
            vt = val_tracks.get((int(vid), int(tid)))
            track_stats.append(
                {
                    "sample_id": sid,
                    "protocol_role": next(x["protocol_role"] for x in label_rows if x["sample_id"] == sid),
                    "num_frames": len(vt["frames"]) if vt else 0,
                    "video_id": int(vid),
                }
            )
        with open(root / "stats" / "track_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "protocol_role", "num_frames", "video_id"])
            w.writeheader()
            w.writerows(track_stats)
        novel_cats = [c for c in roles["novel"] if cat_tracks[c]["tracks"] > 0]
        single = sum(1 for c in novel_cats if cat_tracks[c]["tracks"] == 1)
        cross_video = sum(1 for c in novel_cats if len(cat_tracks[c]["videos"]) >= 2)
        ds = {
            "protocol": proto,
            "version": "TrackOCD-v1.0",
            "known_categories": len(roles["known"]),
            "supported_known_categories": len(roles["known"] - roles["zero_shot_known"]),
            "zero_shot_known_categories": len(roles["zero_shot_known"]),
            "novel_categories_total": len(roles["novel"]),
            "novel_categories_appearing_in_val": len(novel_cats),
            "val_tracks": {k: v for k, v in role_counts.items()},
            "full_tracks": len(full_ids),
            "repeated_tracks": len(repeated_ids),
            "balanced_tracks": len(balanced_ids),
            "repeated_novel_categories": len(set(
                r["ground_truth_category_id"] for r in label_rows if r["sample_id"] in repeated_ids)),
            "balanced_novel_categories": len(set(
                r["ground_truth_category_id"] for r in label_rows if r["sample_id"] in balanced_ids)),
            "novel_singleton_categories": single,
            "novel_cross_video_categories": cross_video,
        }
        (root / "stats" / "dataset_stats.json").write_text(json.dumps(ds, indent=2))
        summary[proto] = ds
        print(json.dumps(ds, indent=1), flush=True)

    (OUT_ROOT / "protocols.json").write_text(json.dumps(summary, indent=2))
    print("protocols built", flush=True)


if __name__ == "__main__":
    main()
