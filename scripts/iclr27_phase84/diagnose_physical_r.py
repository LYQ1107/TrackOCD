#!/usr/bin/env python3
"""Post-hoc, TRAIN-label-only diagnostic for the A84P physical R route."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LINEAGE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/physical/full_temporal_lineage.jsonl")
UNIONS = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs/physical/union_events.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
CSV_PATH = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = ROOT / "outputs/iclr27_phase84/audit/physical_r_diagnostic.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def parse_box(v: Any) -> list[float] | None:
    try:
        x = [float(z) for z in (json.loads(v) if isinstance(v, str) else v)]
        return x if len(x) == 4 else None
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0., x2-x1)*max(0., y2-y1); aa = max(0., a[2]-a[0])*max(0., a[3]-a[1]); bb = max(0., b[2]-b[0])*max(0., b[3]-b[1]); return inter/max(aa+bb-inter, 1e-8)


def main() -> None:
    gt: dict[tuple[int, int], list[tuple[list[float], int]]] = defaultdict(list)
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = parse_box(row.get("gt_bbox_xyxy"));
            if b is not None:
                try: c = int(float(row.get("gt_category_id_common", "-1") or -1))
                except Exception: c = -1
                if c >= 0: gt[(int(row["video_id"]), int(row["image_id"]))].append((b, c))
    unions = [json.loads(line) for line in UNIONS.open(encoding="utf-8") if line.strip()]
    event_lookup = {(int(e["video_id"]), int(e["frame_id"]), int(e["image_id"]), int(e["child_original_physical_track_id"])): e for e in unions}
    native_features = np.asarray(np.load(FEATURES, allow_pickle=False)["features"], dtype=np.float32)
    if not LINEAGE.is_file() or native_features.ndim != 2: raise FileNotFoundError(LINEAGE)
    root_counts: dict[int, Counter[int]] = defaultdict(Counter); original_counts: dict[int, Counter[int]] = defaultdict(Counter); root_sum: dict[int, np.ndarray] = {}; root_n: Counter[int] = Counter(); event_cos: list[float] = []; last_by_original: dict[tuple[int, int], np.ndarray] = {}; row_count = 0; mapped_count = 0
    with LINEAGE.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            row = json.loads(line); row_count += 1; video = int(row["video_id"]); image = int(row.get("image_id", -1)); frame = int(row.get("frame_id", 0)); original = int(row.get("original_physical_track_id", row.get("physical_track_id", -1))); root = int(row.get("phase84_canonical_physical_track_id", row.get("physical_track_id", -1))); feat = native_features[idx]
            cats = gt.get((video, image), []); cat = max(cats, key=lambda x: iou(parse_box(row.get("bbox_xyxy")), x[0]))[1] if cats and parse_box(row.get("bbox_xyxy")) is not None else -1
            if cat >= 0:
                root_counts[root][cat] += 1; original_counts[(video, original)][cat] += 1; mapped_count += 1
            root_sum[root] = root_sum.get(root, np.zeros(768, dtype=np.float64)) + feat.astype(np.float64); root_n[root] += 1
            event = event_lookup.get((video, frame, image, original))
            if event is not None:
                parent = int(event["parent_original_physical_track_id"]); prev = last_by_original.get((video, parent));
                if prev is not None: event_cos.append(float(np.dot(feat, prev) / max(np.linalg.norm(feat)*np.linalg.norm(prev), 1e-8)))
            last_by_original[(video, original)] = feat.copy()
    root_means = {r: (s / max(root_n[r], 1)).astype(np.float32) for r, s in root_sum.items()}; var_sum: Counter[int] = Counter(); var_n: Counter[int] = Counter()
    with LINEAGE.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip(): continue
            row = json.loads(line); root = int(row.get("phase84_canonical_physical_track_id", row.get("physical_track_id", -1))); feat = native_features[idx]; m = root_means[root]; var_sum[root] += 1.0 - float(np.dot(feat, m) / max(np.linalg.norm(feat)*np.linalg.norm(m), 1e-8)); var_n[root] += 1
    merge_counter = Counter(); merge_cos = []
    for e in unions:
        v = int(e["video_id"]); child = (v, int(e["child_original_physical_track_id"])); parent = (v, int(e["parent_original_physical_track_id"])); a = set(original_counts.get(child, {})); b = set(original_counts.get(parent, {}));
        if not a or not b: merge_counter["unlabeled"] += 1
        elif a & b: merge_counter["same_category_overlap"] += 1
        else: merge_counter["cross_category_disjoint"] += 1
        merge_cos.append(float(e.get("score", 0.0)))
    contamination = {"roots_with_train_category": sum(bool(c) for c in root_counts.values()), "roots_multi_category": sum(len(c) > 1 for c in root_counts.values()), "fraction_multi_category": float(sum(len(c) > 1 for c in root_counts.values()) / max(1, sum(bool(c) for c in root_counts.values())))}
    result = {"schema_version": "trackocd.phase84.physical_r_diagnostic.v1", "phase": "Phase84 A84P diagnostic", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "lineage": str(LINEAGE.resolve()), "lineage_sha256": sha256(LINEAGE), "union_events": str(UNIONS.resolve()), "union_events_sha256": sha256(UNIONS), "native_features_sha256": sha256(FEATURES), "rows": row_count, "posthoc_train_gt_row_matches": mapped_count, "union_count": len(unions), "merge_precision_posthoc": {"same_category_overlap": merge_counter["same_category_overlap"], "cross_category_disjoint": merge_counter["cross_category_disjoint"], "unlabeled": merge_counter["unlabeled"], "same_category_fraction_labeled": float(merge_counter["same_category_overlap"] / max(1, merge_counter["same_category_overlap"] + merge_counter["cross_category_disjoint"]))}, "semantic_contamination": contamination, "same_cross_category_merge_rate": {"same": merge_counter["same_category_overlap"], "cross": merge_counter["cross_category_disjoint"]}, "within_root_feature_variance": {"root_count": len(root_means), "mean_one_minus_cosine_to_root_mean": float(sum(var_sum.values()) / max(1, sum(var_n.values()))), "median_one_minus_cosine_to_root_mean": float(np.median([var_sum[r] / max(var_n[r], 1) for r in var_n])) if var_n else 0.0}, "union_similarity": {"appearance_cosine_before_union_count": len(event_cos), "appearance_cosine_before_union_mean": float(np.mean(event_cos)) if event_cos else None, "appearance_cosine_before_union_median": float(np.median(event_cos)) if event_cos else None, "assignment_score_mean": float(np.mean(merge_cos)) if merge_cos else None, "assignment_score_median": float(np.median(merge_cos)) if merge_cos else None, "note": "event appearance cosine is computed against the last observed parent-original feature before each causal union; TRAIN GT is post-hoc diagnostic only"}, "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "labels_used_for_model": False}
    atomic_json(OUT, result); print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
