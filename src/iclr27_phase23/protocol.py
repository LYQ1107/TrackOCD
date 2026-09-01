"""Read-only helpers for the fixed TrackOCD Phase23 proposal protocol.

The helpers deliberately keep the Phase21/22 row keys, causal ordering and
76-event denominator unchanged.  Candidate generation is a proposal-side
diagnostic: each generated box is attached to its parent row and therefore
inherits that row's assigned bit for the reliable-observation rule.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
FEAT_PATH = ROOT / "data/iclr27_phase19r/sources/public_cls_roi.npz"
FEAT_META_PATH = ROOT / "outputs/iclr27_phase15s/features/public_cls_roi.npz.json"
POS_PATH = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
P22_MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"

PREFIXES = (1, 2, 4, 8, 16)
IOU_THRESHOLDS = (0.3, 0.5, 0.7)

# One pre-registered finite transform grid.  A shift is a fraction of the
# source box width/height, making it resolution independent.  The grid is
# intentionally modest (27 boxes per proposal) and is applied identically to
# TRAIN diagnostics and the held-event oracle.
SCALE_FACTORS = (0.80, 1.00, 1.25)
CENTER_SHIFTS = (-0.20, 0.0, 0.20)
MAX_HISTORY = 4


def fval(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default))
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def parse_box(value: str | None) -> list[float] | None:
    try:
        x = [float(v) for v in json.loads(value or "")]
        return x if len(x) == 4 and all(math.isfinite(v) for v in x) else None
    except Exception:
        return None


def normalized_gt(row: dict[str, str]) -> list[float] | None:
    b = parse_box(row.get("gt_bbox_xyxy"))
    w, h = fval(row, "image_width"), fval(row, "image_height")
    if b is None or w <= 0 or h <= 0:
        return None
    return [b[0] / w, b[1] / h, b[2] / w, b[3] / h]


def raw_box(row: dict[str, str]) -> list[float]:
    return [fval(row, k) for k in ("box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm")]


def track_key(row: dict[str, str]) -> str:
    return f"v{int(row['video_id'])}:p{int(row['track_id'])}"


def order_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(row.get("event_rank", 0)), int(row.get("frame_id", 0)), int(row.get("proposal_local_id", 0)))


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(row: dict[str, str]) -> str:
    """Canonical five-field key used by both the corrected CSV and NPZ cache."""
    key = str(row.get("row_key", ""))
    if key:
        return key
    return ":".join(str(row.get(k, "")) for k in
                     ("video_id", "frame_id", "proposal_local_id", "track_id", "image_id"))


def load_aligned_features(rows: list[dict[str, str]] | None = None) -> tuple[Any, Any, dict[str, Any]]:
    """Load frozen features and reorder them to the corrected CSV row order.

    Phase22 indexed the NPZ by positional row index although the NPZ was built
    from the older proposal CSV ordering.  The key set is identical, so an
    in-memory permutation repairs the pairing without copying or rewriting the
    large feature artifact.  The returned report is intentionally explicit so
    callers cannot silently use the positional order.
    """
    import numpy as np
    if rows is None:
        rows = load_rows()
    z = np.load(FEAT_PATH, allow_pickle=False)
    cls, roi = z["cls"], z["roi"]
    feature_keys = [str(x) for x in z["row_keys"]]
    target_keys = [row_key(r) for r in rows]
    fmap = {k: i for i, k in enumerate(feature_keys)}
    missing = [k for k in target_keys if k not in fmap]
    extra = [k for k in feature_keys if k not in set(target_keys)]
    if missing or extra or len(set(target_keys)) != len(target_keys) or len(set(feature_keys)) != len(feature_keys):
        raise RuntimeError(f"feature/row key set mismatch: missing={len(missing)} extra={len(extra)} duplicate_csv={len(target_keys)-len(set(target_keys))} duplicate_feat={len(feature_keys)-len(set(feature_keys))}")
    permutation = np.asarray([fmap[k] for k in target_keys], dtype=np.int64)
    aligned = {
        "cls": cls[permutation],
        "roi": roi[permutation],
    }
    report = {
        "feature_path": str(FEAT_PATH),
        "feature_meta_path": str(FEAT_META_PATH),
        "feature_rows": len(feature_keys),
        "csv_rows": len(target_keys),
        "positional_match_count": int(sum(a == b for a, b in zip(target_keys, feature_keys))),
        "set_overlap_count": int(len(set(target_keys) & set(feature_keys))),
        "aligned_exact_count": int(sum(target_keys[i] == feature_keys[permutation[i]] for i in range(len(target_keys)))),
        "permutation_sha256": __import__("hashlib").sha256(permutation.tobytes()).hexdigest(),
        "reordered_in_memory": True,
        "source_order_is_not_used": True,
    }
    return aligned["cls"], aligned["roi"], report


def load_events() -> list[dict[str, Any]]:
    events = [json.loads(x) for x in POS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(events) != 76:
        raise RuntimeError(f"positive event denominator changed: {len(events)}")
    return sorted(events, key=lambda x: str(x["event_key"]))


def by_track(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        out[track_key(row)].append(i)
    for key in out:
        out[key].sort(key=lambda i: order_key(rows[i]))
    return out


def track_positions(rows: list[dict[str, str]], tracks: dict[str, list[int]] | None = None) -> dict[int, int]:
    """Return O(1) positions for causal history lookup."""
    tracks = tracks or by_track(rows)
    return {idx: pos for inds in tracks.values() for pos, idx in enumerate(inds)}


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    aa = list(a); bb = list(b)
    x1, y1 = max(aa[0], bb[0]), max(aa[1], bb[1])
    x2, y2 = min(aa[2], bb[2]), min(aa[3], bb[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, aa[2] - aa[0]) * max(0.0, aa[3] - aa[1])
    area_b = max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def transform_box(box: Iterable[float], scale: float, dx: float, dy: float) -> list[float]:
    b = list(box)
    cx, cy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    bw, bh = max(0.0, b[2] - b[0]), max(0.0, b[3] - b[1])
    cx += dx * bw; cy += dy * bh
    nw, nh = bw * scale, bh * scale
    return [max(0.0, min(1.0, cx - nw * 0.5)), max(0.0, min(1.0, cy - nh * 0.5)),
            max(0.0, min(1.0, cx + nw * 0.5)), max(0.0, min(1.0, cy + nh * 0.5))]


def fixed_transforms(box: Iterable[float]) -> list[list[float]]:
    return [transform_box(box, scale, dx, dy) for scale in SCALE_FACTORS for dx in CENTER_SHIFTS for dy in CENTER_SHIFTS]


def row_candidates(rows: list[dict[str, str]], idx: int, track_indices: dict[str, list[int]], positions: dict[int, int] | None = None) -> list[dict[str, Any]]:
    """Return current-row transforms plus causal same-track history boxes."""
    row = rows[idx]; inds = track_indices[track_key(row)]
    pos = positions[idx] if positions is not None else inds.index(idx)
    hist = inds[max(0, pos - MAX_HISTORY + 1):pos + 1]
    out: list[dict[str, Any]] = []
    for source_idx in hist:
        source = rows[source_idx]
        for transform_id, candidate in enumerate(fixed_transforms(raw_box(source))):
            out.append({"parent_index": source_idx, "source_frame": int(source["frame_id"]), "transform_id": transform_id,
                        "box": candidate, "assigned": str(source.get("assigned", "0")) == "1"})
    return out


def event_indices(rows: list[dict[str, str]], tracks: dict[str, list[int]], event: dict[str, Any], prefix: int) -> tuple[list[int], list[int]]:
    sk, tk = str(event["source_tracklet_keys"][0]), str(event["target_tracklet_key"])
    source = tracks.get(sk, [])
    target = tracks.get(tk, [])[:min(prefix, len(tracks.get(tk, [])))]
    return source, target
