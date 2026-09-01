#!/usr/bin/env python3
"""Build Phase25 lightweight candidate-set indices.

Only row/parent indices, causal geometry, assignment bits and TRAIN IoU
targets are stored.  The frozen DINOv2 feature cache is gathered by parent
index at training time and is never copied.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase25.protocol import (
    CSV_PATH, FEAT_PATH, P22_MANIFEST, GEOM_FIELDS, TRANSFORM_META,
    by_track, candidate_arrays, fval, load_aligned_features, normalized_gt,
    track_positions,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase25/manifests"
MAX_CANDIDATES = 108


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def iou_vec(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1]); x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1); aa = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1]); ab = max(0., gt[2] - gt[0]) * max(0., gt[3] - gt[1]); return inter / np.maximum(aa + ab - inter, 1e-8)


def build_split(rows: list[dict[str, str]], indices: list[int], tracks: dict[str, list[int]], positions: dict[int, int], path: Path) -> dict[str, Any]:
    selected = [i for i in indices if normalized_gt(rows[i]) is not None]
    n, m = len(selected), MAX_CANDIDATES
    parent = np.full((n, m), -1, np.int32); geom = np.zeros((n, m, len(GEOM_FIELDS) + 4 + 3), np.float32); labels = np.zeros((n, m), np.float32); assigned = np.zeros((n, m), bool); mask = np.zeros((n, m), bool); boxes_out = np.zeros((n, m, 4), np.float32); trans_out = np.full((n, m), -1, np.int16); row_idx = np.asarray(selected, np.int32); row_category = np.asarray([int(rows[i].get("gt_category_id_common", -1)) for i in selected], np.int32); row_video = np.asarray([int(rows[i].get("video_id", -1)) for i in selected], np.int32); counts = []
    for k, idx in enumerate(selected):
        boxes, pp, trans, ass = candidate_arrays(rows, idx, tracks, positions)
        if len(boxes) > m: raise RuntimeError(f"candidate count {len(boxes)} exceeds fixed bound {m}")
        gt = np.asarray(normalized_gt(rows[idx]), np.float32); lab = iou_vec(boxes, gt)
        parent[k, :len(pp)] = pp; boxes_out[k, :len(pp)] = boxes; trans_out[k, :len(pp)] = trans; labels[k, :len(pp)] = lab; assigned[k, :len(pp)] = ass; mask[k, :len(pp)] = True
        for j, p in enumerate(pp.tolist()): geom[k, j, :len(GEOM_FIELDS)] = [fval(rows[int(p)], q) for q in GEOM_FIELDS]
        geom[k, :len(pp), len(GEOM_FIELDS):len(GEOM_FIELDS) + 4] = boxes; geom[k, :len(pp), len(GEOM_FIELDS) + 4:] = TRANSFORM_META[trans]; counts.append(len(pp))
    atomic_npz(path, row_idx=row_idx, parent_idx=parent, geom=geom, label_iou=labels, parent_assigned=assigned, mask=mask, candidate_box=boxes_out, transform_id=trans_out, row_category=row_category, row_video=row_video)
    keys = "\n".join(str(rows[i].get("row_key", "")) for i in selected).encode(); return {"path": str(path), "rows": n, "candidate_slots": int(mask.sum()), "max_candidates": int(max(counts, default=0)), "mean_candidates": float(np.mean(counts)) if counts else 0., "row_key_sha256": hashlib.sha256(keys).hexdigest(), "sha256": sha256(path)}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); load_aligned_features(rows)  # explicit key-set smoke; features remain external
    manifest = json.loads(P22_MANIFEST.read_text(encoding="utf-8")); tracks = by_track(rows); positions = track_positions(rows, tracks); OUT.mkdir(parents=True, exist_ok=True); result: dict[str, Any] = {"protocol": "trackocd_iclr27_phase25_set_aware_candidate_manifest", "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "feature_path": str(FEAT_PATH), "feature_artifact_not_copied": True, "max_candidates": MAX_CANDIDATES, "geometry_fields": list(GEOM_FIELDS), "folds": []}
    for f in manifest["folds"]:
        fold = int(f["fold"]); fit_v, val_v = set(map(int, f["fit_videos"])), set(map(int, f["validation_videos"])); fit_c, held_c = set(map(int, f["fit_categories"])), set(map(int, f["held_categories"])); fit_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in fit_v and int(r.get("gt_category_id_common", -1)) in fit_c]; val_idx = [i for i, r in enumerate(rows) if int(r["video_id"]) in val_v and int(r.get("gt_category_id_common", -1)) in held_c]; fit = build_split(rows, fit_idx, tracks, positions, OUT / f"setaware_fit_f{fold}.npz"); val = build_split(rows, val_idx, tracks, positions, OUT / f"setaware_val_f{fold}.npz"); result["folds"].append({"fold": fold, "fit_videos": len(fit_v), "validation_videos": len(val_v), "fit_categories": len(fit_c), "held_categories": len(held_c), "fit": fit, "validation": val})
    fd, tmp = tempfile.mkstemp(prefix=".setaware_manifest.", dir=str(OUT))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(result, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, OUT / "setaware_manifest.json")
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    atomic = OUT.parent / "completion/manifests.done"; atomic.parent.mkdir(parents=True, exist_ok=True); atomic.write_text(json.dumps({"stage": "phase25_set_manifests", "folds": len(result["folds"]), "feature_artifact_not_copied": True}) + "\n", encoding="utf-8")
    print(json.dumps({"folds": [{"fold": x["fold"], "fit_rows": x["fit"]["rows"], "val_rows": x["validation"]["rows"], "fit_candidates": x["fit"]["candidate_slots"], "val_candidates": x["validation"]["candidate_slots"]} for x in result["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
