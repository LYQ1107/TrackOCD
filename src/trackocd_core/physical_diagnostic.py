"""Geometry-only clipped-stream diagnostics; labels enter post-seal evaluation."""
from __future__ import annotations
from collections import Counter, defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def metadata_prefixes(annotation: dict, videos: int = 4, frames: int = 16) -> list[dict]:
    """Never inspect annotation/category/track labels when selecting images."""
    ids = sorted(int(v["id"]) for v in annotation["videos"])[:videos]
    rows = []
    for video_id in ids:
        images = sorted((im for im in annotation["images"] if int(im["video_id"]) == video_id),
                        key=lambda im: (int(im["frame_index"]), int(im["id"])))[:frames]
        if not images:
            raise ValueError("Selected video has no image metadata; do not substitute another")
        rows.append({"video_id": video_id, "images": [{"image_id": int(im["id"]), "frame_index": int(im["frame_index"]),
                                                       "image_path": im["file_name"]} for im in images]})
    return rows


def pairwise_iou(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    gt, pred = np.asarray(gt, dtype=float).reshape(-1, 4), np.asarray(pred, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(gt[:, None, 2:], pred[None, :, 2:]) - np.maximum(gt[:, None, :2], pred[None, :, :2])).prod(axis=2)
    union = np.maximum(0, gt[:, 2:] - gt[:, :2]).prod(axis=1)[:, None] + np.maximum(0, pred[:, 2:] - pred[:, :2]).prod(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def observed_purity(frames: list[dict], threshold: float = .5) -> dict:
    """Matched category/identity mixing is observed; unmatched is not background."""
    counts, categories, targets = Counter(), defaultdict(Counter), defaultdict(set)
    matched_rows = 0
    for frame in frames:
        ids = np.asarray(frame["pred_ids"], dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Duplicate physical identity in a frame")
        counts.update(ids.tolist())
        similarity = pairwise_iou(frame["gt_boxes"], frame["pred_boxes"])
        if similarity.size:
            bonus = max(similarity.shape) + 1
            weights = np.where(similarity >= threshold, bonus + similarity, 0)
            gt_rows, pred_rows = linear_sum_assignment(weights, maximize=True)
            for gt, pred in zip(gt_rows, pred_rows):
                if weights[gt, pred] > 0:
                    local_id = int(ids[pred])
                    categories[local_id][int(frame["gt_categories"][gt])] += 1
                    targets[local_id].add(int(frame["gt_ids"][gt]))
                    matched_rows += 1
    mixed_category = sum(len(categories[i]) > 1 for i in counts)
    multiple_identity = sum(len(targets[i]) > 1 for i in counts)
    unsupported = sum(not categories[i] for i in counts)
    unknown_member_tracks = sum(sum(categories[i].values()) < counts[i] for i in counts)
    majority = sum(max(categories[i].values(), default=0) for i in counts)
    return {"physical_tracks": len(counts), "prediction_rows": sum(counts.values()), "matched_rows": matched_rows,
            "unmatched_unknown_rows": sum(counts.values()) - matched_rows,
            "observed_multicategory_tracks": mixed_category, "observed_multiple_gt_identity_tracks": multiple_identity,
            "tracks_with_no_gt_match": unsupported, "tracks_with_unknown_observations": unknown_member_tracks,
            "observed_majority_category_fraction_over_matched_rows": majority / matched_rows if matched_rows else None,
            "entire_observed_track_matched_to_one_category": sum(len(categories[i]) == 1 and sum(categories[i].values()) == counts[i] for i in counts),
            "incomplete_annotation_boundary": "Unmatched rows are unknown, not proven false positives/background; single observed category is not global semantic purity",
            "matching": "Per-frame class-free maximum-cardinality then IoU Hungarian, IoU>=0.5; labels only aggregate afterward"}
