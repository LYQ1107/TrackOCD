"""Read-only Phase26 facade and causal candidate-source helpers."""
from __future__ import annotations

import numpy as np

from src.iclr27_phase24.protocol import (  # noqa: F401
    CSV_PATH, FEAT_PATH, FEAT_META_PATH, POS_PATH, P22_MANIFEST,
    PREFIXES, IOU_THRESHOLDS, SCALE_FACTORS, CENTER_SHIFTS, MAX_HISTORY,
    by_track, fval, load_aligned_features, load_events, normalized_gt,
    raw_box, track_positions, track_key, order_key,
)
from src.iclr27_phase23.protocol import row_key

def load_aligned_features(rows=None):
    """Efficient key-aligned read of the frozen feature NPZ.

    The inherited helper rebuilt a 43k-element set inside a list
    comprehension; Phase26 keeps the same permutation semantics with one
    cached set and never rewrites the source artifact.
    """
    import hashlib
    if rows is None:
        import csv
        rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    z = np.load(FEAT_PATH, allow_pickle=False)
    cls, roi = z["cls"], z["roi"]
    feature_keys = [str(x) for x in z["row_keys"]]
    target_keys = [row_key(r) for r in rows]
    fmap = {k: i for i, k in enumerate(feature_keys)}; target_set = set(target_keys); feature_set = set(feature_keys)
    missing = [k for k in target_keys if k not in fmap]; extra = [k for k in feature_keys if k not in target_set]
    if missing or extra or len(target_set) != len(target_keys) or len(feature_set) != len(feature_keys):
        raise RuntimeError(f"feature/row key set mismatch: missing={len(missing)} extra={len(extra)}")
    perm = np.asarray([fmap[k] for k in target_keys], dtype=np.int64)
    report = {"feature_path": str(FEAT_PATH), "feature_rows": len(feature_keys), "csv_rows": len(target_keys), "positional_match_count": int(sum(a == b for a, b in zip(target_keys, feature_keys))), "set_overlap_count": len(target_set & feature_set), "aligned_exact_count": int(sum(target_keys[i] == feature_keys[perm[i]] for i in range(len(target_keys)))), "permutation_sha256": hashlib.sha256(perm.tobytes()).hexdigest(), "reordered_in_memory": True, "source_order_is_not_used": True}
    return cls[perm], roi[perm], report
from scripts.iclr27_phase23.train_quality_ranker import (  # noqa: F401
    TRANSFORM_META, candidate_arrays, feature_batch, iou_vec,
)

PROTOCOL = "trackocd_iclr27_phase26_proposal_source_candidate_coverage"
RELIABLE_RULE = "parent assigned == 1 and transformed true normalized IoU >= 0.5"
TOP_KS = (5, 10, 20, 27)

# Additional source grid is fixed before held-event evaluation.  It is causal,
# resolution independent and uses only a row's box/history; it is diagnostic.
BROAD_SCALES = (0.55, 1.00, 1.45)
BROAD_SHIFTS = (-0.35, -0.12, 0.12, 0.35)
SOURCE_ANCHORS = np.asarray([
    [1.00, 0.00, 0.00], [0.70, 0.00, 0.00], [1.40, 0.00, 0.00],
    [1.00, -0.30, 0.00], [1.00, 0.30, 0.00],
    [1.00, 0.00, -0.30], [1.00, 0.00, 0.30], [1.35, 0.22, -0.22],
], dtype=np.float32)
GEOM_FIELDS = (
    "score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm",
    "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm",
    "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou",
)


def transform_box(box: np.ndarray, scale: float, dx: float, dy: float) -> np.ndarray:
    b = np.asarray(box, dtype=np.float32)
    cx = (b[0] + b[2]) * .5 + dx * max(0., b[2] - b[0])
    cy = (b[1] + b[3]) * .5 + dy * max(0., b[3] - b[1])
    w = max(0., b[2] - b[0]) * scale; h = max(0., b[3] - b[1]) * scale
    return np.clip(np.asarray([cx - .5*w, cy - .5*h, cx + .5*w, cy + .5*h], np.float32), 0., 1.)


def broad_candidates(rows: list[dict[str, str]], idx: int,
                     tracks: dict[str, list[int]], positions: dict[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the fixed 48-transform/history source extension."""
    r = rows[idx]; inds = tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"]
    pos = positions[idx]; hist = inds[max(0, pos - MAX_HISTORY + 1):pos + 1]
    boxes = []; parents = []; trans = []; assigned = []
    for p in hist:
        base = np.asarray(raw_box(rows[p]), np.float32)
        for tid, s in enumerate(BROAD_SCALES):
            for dx in BROAD_SHIFTS:
                for dy in BROAD_SHIFTS:
                    boxes.append(transform_box(base, s, dx, dy)); parents.append(p); trans.append(tid); assigned.append(str(rows[p].get("assigned", "0")) == "1")
    return np.asarray(boxes, np.float32), np.asarray(parents, np.int32), np.asarray(trans, np.int16), np.asarray(assigned, bool)


def source_anchor_boxes(box: np.ndarray) -> np.ndarray:
    return np.asarray([transform_box(box, float(a[0]), float(a[1]), float(a[2])) for a in SOURCE_ANCHORS], np.float32)


def iou_np(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    b = np.asarray(boxes, np.float32); g = np.asarray(gt, np.float32)
    x1 = np.maximum(b[:, 0], g[0]); y1 = np.maximum(b[:, 1], g[1]); x2 = np.minimum(b[:, 2], g[2]); y2 = np.minimum(b[:, 3], g[3])
    inter = np.maximum(0., x2-x1) * np.maximum(0., y2-y1)
    aa = np.maximum(0., b[:, 2]-b[:, 0]) * np.maximum(0., b[:, 3]-b[:, 1]); ag = max(0., g[2]-g[0]) * max(0., g[3]-g[1])
    return inter / np.maximum(aa + ag - inter, 1e-8)
