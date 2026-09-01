"""Build the preregistered real-video TrackOCD-DEV+ manifest.

The split is selected from public TAO TRAIN annotation counts only.  Existing
DINOv2 embeddings are used only to record coverage and to make a compact,
label-free diagnostic cache; they never influence the category/video split.
Q1 is not read by this script.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/OCD_OVMOT")
ANN = ROOT / "data/iclr27_phase14b/sources/tao_train_annotations.json"
FULL_FEATURES = ROOT / "data/iclr27_phase14b/sources/full_tao_tracks.npz"
KNOWN_IDS = ROOT / "data/trackocd_v1/pure/splits/supported_known_ids.json"
FRAME_ROOT = ROOT / "data/iclr27_phase14b/sources/tao_train_frames"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def pair_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cat[int(row["category_id"])].append(row)
    same = cross_video = 0
    for tracks in by_cat.values():
        for i in range(len(tracks)):
            for j in range(i):
                same += 1
                if tracks[i]["video_id"] != tracks[j]["video_id"]:
                    cross_video += 1
    return {
        "tracks": len(rows),
        "categories": len(by_cat),
        "videos": len({r["video_id"] for r in rows}),
        "same_category_cross_physical_pairs": same,
        "same_category_cross_video_pairs": cross_video,
    }


def load_tracks() -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    data = json.loads(ANN.read_text())
    images = {int(im["id"]): im for im in data["images"]}
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd"):
            continue
        groups[(int(ann["video_id"]), int(ann["track_id"]))].append(ann)
    tracks: list[dict[str, Any]] = []
    for (video_id, track_id), anns in groups.items():
        anns.sort(key=lambda a: (int(images[int(a["image_id"])] ["frame_index"]), int(a["image_id"])))
        first_category = int(anns[0]["category_id"])
        paths: list[str] = []
        boxes: list[list[float]] = []
        frames: list[int] = []
        for ann in anns:
            image = images[int(ann["image_id"])]
            x, y, w, h = [float(v) for v in ann["bbox"]]
            paths.append(str(image["file_name"]))
            boxes.append([x, y, x + w, y + h])
            frames.append(int(image["frame_index"]))
        tracks.append({
            "sample_id": f"{video_id}_{track_id}",
            "source_dataset": "TAO TRAIN",
            "video_id": video_id,
            "track_id": track_id,
            "category_id": first_category,
            "frame_indices": frames,
            "image_paths": paths,
            "boxes_xyxy": boxes,
            "num_frames": len(paths),
            "source_frame_root": str(FRAME_ROOT),
        })
    videos = {int(v["id"]): v for v in data["videos"]}
    return tracks, videos


def choose_categories(tracks: list[dict[str, Any]], known: set[int]) -> list[int]:
    by_cat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in tracks:
        if int(row["category_id"]) not in known:
            by_cat[int(row["category_id"])].append(row)
    eligible: list[tuple[int, int, int, int]] = []
    for category_id, rows in by_cat.items():
        videos = {int(r["video_id"]) for r in rows}
        cross_pairs = sum(
            1
            for i in range(len(rows))
            for j in range(i)
            if int(rows[i]["video_id"]) != int(rows[j]["video_id"])
        )
        if len(rows) >= 3 and len(videos) >= 2:
            eligible.append((cross_pairs, len(rows), category_id, len(videos)))
    eligible.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[2] for x in eligible[:20]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/iclr27_phase14b")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    tracks, videos = load_tracks()
    known = {int(x) for x in json.loads(KNOWN_IDS.read_text())}
    dev_categories = choose_categories(tracks, known)
    dev_videos = {
        int(row["video_id"])
        for row in tracks
        if int(row["category_id"]) in set(dev_categories)
    }
    remaining_videos = sorted(set(videos) - dev_videos)
    calibration_videos = {
        video_id for index, video_id in enumerate(remaining_videos) if index % 10 == 0
    }

    full = np.load(FULL_FEATURES, allow_pickle=False)
    feature_ids = [str(x) for x in full["sample_ids"]]
    feature_index = {sid: i for i, sid in enumerate(feature_ids)}

    dev_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    for row in tracks:
        category = int(row["category_id"])
        video_id = int(row["video_id"])
        if video_id in dev_videos:
            if category in dev_categories:
                role = "devplus_novel"
                split = "devplus"
                proposal_alignment = "gt_box_diagnostic_only"
                target = dev_rows
            else:
                role = "excluded_video_overlap"
                split = "excluded"
                proposal_alignment = "not_used"
                target = excluded_rows
        elif video_id in calibration_videos:
            role = "calibration_public_train"
            split = "calibration"
            proposal_alignment = "gt_box_diagnostic_only"
            target = calibration_rows
        else:
            role = "representation_train_public"
            split = "representation_train"
            proposal_alignment = "gt_box_diagnostic_only"
            target = train_rows
        item = dict(row)
        item.update({
            "split": split,
            "role": role,
            "known_novel_evaluation_role": (
                "novel" if role == "devplus_novel" else "not_evaluated"
            ),
            "proposal_alignment": proposal_alignment,
            "category_label_access": "offline_split_builder_only",
            "q1_label_used": False,
            "private_gt_used": False,
            "future_frames_used": False,
            "physical_id_used_as_feature": False,
            "chronological_position": None,
            "feature_row_available": row["sample_id"] in feature_index,
        })
        target.append(item)

    # The public stream order is deterministic and causal.  Position counts
    # tracks, not frames, because the evaluator consumes a track stream.
    ordered_dev = sorted(dev_rows, key=lambda r: (
        int(r["video_id"]), int(r["frame_indices"][0]), int(r["track_id"]), r["sample_id"]
    ))
    for position, row in enumerate(ordered_dev):
        row["chronological_position"] = position

    dev_stats = pair_counts(ordered_dev)
    train_stats = pair_counts(train_rows)
    calibration_stats = pair_counts(calibration_rows)
    all_stats = pair_counts(tracks)
    missing_dev = [r["sample_id"] for r in ordered_dev if not r["feature_row_available"]]

    # Label-free canonical diagnostic cache.  Labels remain only in the
    # manifest/evaluator sidecar; feature arrays contain embeddings, masks and
    # opaque sample keys only.
    dev_indices = [feature_index[r["sample_id"]] for r in ordered_dev if r["feature_row_available"]]
    dev_feature_ids = [r["sample_id"] for r in ordered_dev if r["feature_row_available"]]
    atomic_npz(
        out / "features" / "devplus_dinov2_gtbox.npz",
        sample_keys=np.asarray(dev_feature_ids),
        frame_features=full["frame_feats"][dev_indices],
        frame_mask=full["frame_mask"][dev_indices],
        mean_features=full["mean_feats"][dev_indices],
    )

    manifest_rows = ordered_dev + sorted(
        train_rows + calibration_rows + excluded_rows,
        key=lambda r: (r["split"], int(r["video_id"]), int(r["track_id"]), r["sample_id"]),
    )
    atomic_jsonl(out / "manifests" / "devplus_tracks.jsonl", manifest_rows)
    atomic_json(out / "manifests" / "devplus_split.json", {
        "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
        "source_annotation": str(ANN),
        "source_frames": str(FRAME_ROOT),
        "known_ids_source": str(KNOWN_IDS),
        "selected_devplus_categories": dev_categories,
        "devplus_videos": sorted(dev_videos),
        "calibration_videos": sorted(calibration_videos),
        "representation_train_videos": sorted(set(videos) - dev_videos - calibration_videos),
        "q1_used": False,
        "feature_selection_used": False,
        "model_selection_used": False,
    })
    atomic_json(out / "eval" / "opportunity_audit.json", {
        "protocol": "docs/iclr27_phase14b/PROTOCOL.md",
        "selection_rule": "annotation-only top-20 eligible categories by cross-video pairs, track count, category id",
        "targets": {
            "repeated_novel_categories": 20,
            "cross_physical_same_category_pairs": 100,
            "cross_video_same_category_pairs": 30,
        },
        "full_tao_train": all_stats,
        "devplus_gt_box_diagnostic": dev_stats,
        "representation_train": train_stats,
        "calibration": calibration_stats,
        "devplus_feature_coverage": {
            "available_tracks": len(dev_feature_ids),
            "total_tracks": len(ordered_dev),
            "missing_sample_ids": missing_dev,
        },
        "proposal_view": {
            "status": "not_yet_built",
            "reason": "a compatible TAO TRAIN detector/tracker proposal stream must be audited separately; GT boxes are diagnostic only",
        },
        "q1_used": False,
        "private_labels_used_for_split_only": True,
    })
    atomic_json(out / "manifests" / "devplus_meta.json", {
        "source_dataset": "TAO TRAIN (real YFCC100M/BDD/YouTube-VOS videos)",
        "tracks_total": len(tracks),
        "videos_total": len(videos),
        "categories_total": len({int(r["category_id"]) for r in tracks}),
        "selected_devplus_categories": dev_categories,
        "selected_devplus_videos": len(dev_videos),
        "devplus_tracks": len(ordered_dev),
        "calibration_videos": len(calibration_videos),
        "representation_train_videos": len(set(videos) - dev_videos - calibration_videos),
        "missing_devplus_feature_tracks": missing_dev,
        "feature_cache": str(out / "features" / "devplus_dinov2_gtbox.npz"),
        "q1_used": False,
        "future_frames_used": False,
        "physical_id_used_as_feature": False,
    })
    print(json.dumps({"devplus": dev_stats, "selected_categories": dev_categories, "missing": missing_dev}, indent=2))


if __name__ == "__main__":
    main()
