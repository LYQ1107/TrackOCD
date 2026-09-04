#!/usr/bin/env python3
"""Build the registered B84S-Q source/query-conditioned TRAIN manifest.

Unlike the original B84S manifest, every group is tied to one source track from
the Phase30 TRAIN episode contract.  Native Q0 candidates are unchanged; GT
category/IoU fields are used only to make the listwise target.
"""
from __future__ import annotations

import ast
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key

BASE = Path("/data2/usr_for_deadline/trackocd_phase84/project_outputs")
NATIVE_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURE_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
B4_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz")
OBS_PATH = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
SOURCE_CACHE = BASE / "manifests/source_track_native_vectors.npz"
EPISODE_DIR = ROOT / "outputs/iclr27_phase30/manifests"
PUBLIC_CSV = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
OUT = ROOT / "outputs/iclr27_phase84"
DATA = BASE / "manifests/b84sq_query_features.npz"
MANIFEST = OUT / "manifests/b84sq_query_manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def box(value: Any) -> list[float] | None:
    if value is None or value == "":
        return None
    try:
        x = [float(v) for v in (value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value)))]
        return x if len(x) == 4 else None
    except Exception:
        return None


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def main() -> None:
    table = load_frozen_tracks()
    public = list(csv.DictReader(PUBLIC_CSV.open(newline="", encoding="utf-8")))
    native = [json.loads(line) for line in NATIVE_PATH.open(encoding="utf-8") if line.strip()]
    native_feat = np.asarray(np.load(FEATURE_PATH, allow_pickle=False)["features"], dtype=np.float32)
    native_feat /= np.maximum(np.linalg.norm(native_feat, axis=1, keepdims=True), 1e-8)
    if len(native) != len(native_feat):
        raise RuntimeError(f"native/features mismatch {len(native)} vs {len(native_feat)}")
    b4 = np.load(B4_PATH, allow_pickle=False)
    descriptors = np.zeros((len(native), 15), dtype=np.float32)
    descriptors[b4["flat_indices"].astype(np.int64)] = b4["features"].astype(np.float32)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(native):
        if box(row.get("bbox_xyxy")) is not None:
            groups[(int(row["video_id"]), int(row.get("image_id", -1)))].append(i)
    for key in groups:
        groups[key].sort(key=lambda i: (int(native[i].get("candidate_rank") or 0), int(native[i].get("physical_track_id", -1)), i))

    # Public proposals are used only to deterministically locate a source
    # vector and the causal target image; GT is retained for labels below.
    gt_by_image: dict[tuple[int, int], list[tuple[list[float], int]]] = defaultdict(list)
    public_by_track: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in public:
        k = track_key(row)
        public_by_track[k].append(row)
        gb = box(row.get("gt_bbox_xyxy"))
        if gb is not None:
            gt_by_image[(int(row["video_id"]), int(row["image_id"]))].append((gb, int(float(row.get("gt_category_id_common", -1) or -1))))
    for k in public_by_track:
        public_by_track[k].sort(key=order_key)
    by_image = groups
    mapped_by_track: dict[str, list[int]] = defaultdict(list)
    mapped_rows = 0
    for k, rows in public_by_track.items():
        for row in rows:
            pb = box(row.get("bbox_xyxy")); inds = by_image.get((int(row["video_id"]), int(row["image_id"])), [])
            if pb is None or not inds:
                continue
            j = max(inds, key=lambda q: (iou(pb, box(native[q].get("bbox_xyxy"))), float(native[q].get("base_score", 0.0) or 0.0), -int(native[q].get("candidate_rank") or 0), -q))
            if iou(pb, box(native[j].get("bbox_xyxy"))) >= 0.5:
                mapped_by_track[k].append(int(j)); mapped_rows += 1
    source_z = np.load(SOURCE_CACHE, allow_pickle=False)
    source_keys = [str(x) for x in source_z["keys"].tolist()]
    source_idx = {k: i for i, k in enumerate(source_keys)}
    source_v = source_z["vectors"].astype(np.float32)
    source_p = source_z["prototypes"].astype(np.float32)

    blocked_videos: set[int] = set()
    for line in OBS_PATH.open(encoding="utf-8"):
        if line.strip():
            e = json.loads(line); blocked_videos.update({int(e.get("source_video", -1)), int(e.get("target_video", -1))})
    # Keep at most two positive and two hard-DEFER target groups per source in
    # each fixed Phase30 fold/split.  Ordering is deterministic and independent
    # of any held-event result.
    candidates: list[dict[str, Any]] = []
    for fi in range(4):
        ep = json.loads((EPISODE_DIR / f"episode_manifest_f{fi}.json").read_text(encoding="utf-8"))
        for rec in ep["records"]:
            split, kind = str(rec.get("split")), str(rec.get("kind"))
            if split not in {"fit", "val"} or kind not in {"multi_positive_cross_video", "null_no_match_hard_negative"}:
                continue
            q = str(rec.get("query_track_key")); qmeta = table.metadata.get(q)
            if qmeta is None:
                continue
            qv, qcat = int(qmeta["video"]), int(qmeta["category"])
            if qv in blocked_videos:
                continue
            supports = [str(s) for s in rec.get("support_track_keys", [])]
            if not supports:
                continue
            for s in sorted(supports):
                smeta = table.metadata.get(s)
                if smeta is None or int(smeta["video"]) == qv or int(smeta["video"]) in blocked_videos:
                    continue
                if s not in source_idx or not mapped_by_track.get(s):
                    continue
                rows = public_by_track.get(q, [])
                if not rows:
                    continue
                target_row = rows[min(15, len(rows) - 1)]
                image_key = (int(target_row["video_id"]), int(target_row["image_id"]))
                if not groups.get(image_key):
                    continue
                candidates.append({"fold": fi, "split": split, "kind": kind, "episode_id": str(rec.get("episode_id")), "source_key": s, "query_key": q, "source_category": int(smeta["category"]), "query_category": qcat, "target_video": qv, "target_image": image_key[1], "target_row_key": str(target_row.get("row_key", ""))})
    candidates.sort(key=lambda x: (x["fold"], x["split"], x["source_key"], x["kind"], x["episode_id"], x["query_key"]))
    selected: list[dict[str, Any]] = []; counts: defaultdict[tuple[int, str, str, str], int] = defaultdict(int)
    for c in candidates:
        typ = "positive" if c["kind"] == "multi_positive_cross_video" else "defer"
        key = (int(c["fold"]), str(c["split"]), str(c["source_key"]), typ)
        if counts[key] >= 2:
            continue
        counts[key] += 1; selected.append(c)

    features: list[list[float]] = []; offsets = [0]; targets: list[int] = []; videos: list[int] = []; categories: list[int] = []; source_names: list[str] = []; query_names: list[str] = []; folds: dict[str, dict[str, list[int]]] = {str(f): {"fit_groups": [], "validation_groups": []} for f in range(4)}
    cache: dict[tuple[int, int, int], tuple[list[int], np.ndarray]] = {}
    for c in selected:
        gkey = (int(c["target_video"]), int(c["target_image"])); inds = groups[gkey]
        base = descriptors[np.asarray(inds, dtype=np.int64)]
        sv = norm(source_v[4, source_idx[c["source_key"]]])
        ps = [norm(x) for x in source_p[:, source_idx[c["source_key"]]] if float(np.linalg.norm(x)) > 1e-8] or [sv]
        z = native_feat[np.asarray(inds, dtype=np.int64)]
        sims = z @ sv; pmat = np.stack([z @ p for p in ps], axis=1)
        extra = np.stack([sims, pmat.max(1), pmat.mean(1), pmat.min(1)], axis=1).astype(np.float32)
        x = np.concatenate([base, extra], axis=1)
        target = len(inds)
        if c["kind"] == "multi_positive_cross_video":
            gtvals = gt_by_image.get(gkey, [])
            valid = []
            for j, ni in enumerate(inds):
                cb = box(native[ni].get("bbox_xyxy")); best_i, best_cat = max(((iou(cb, gb), cat) for gb, cat in gtvals), default=(0.0, -1))
                if best_i >= 0.5 and best_cat == int(c["source_category"]):
                    valid.append((j, best_i, float(native[ni].get("base_score", 0.0) or 0.0), -int(native[ni].get("candidate_rank") or 0)))
            if valid:
                target = int(max(valid, key=lambda y: (y[1], y[2], y[3]))[0])
        start = len(features); features.extend(x.tolist()); offsets.append(len(features)); targets.append(target); videos.append(int(c["target_video"])); categories.append(int(c["query_category"])); source_names.append(str(c["source_key"])); query_names.append(str(c["query_key"])); folds[str(c["fold"])]["fit_groups" if c["split"] == "fit" else "validation_groups"].append(len(targets) - 1)
    if not features:
        raise RuntimeError("no query-conditioned groups were materialized")
    DATA.parent.mkdir(parents=True, exist_ok=True); tmp = DATA.with_name("." + DATA.name + ".tmp.npz")
    np.savez(tmp, features=np.asarray(features, np.float32), offsets=np.asarray(offsets, np.int64), targets=np.asarray(targets, np.int64), videos=np.asarray(videos, np.int64), categories=np.asarray(categories, np.int64), source_keys=np.asarray(source_names), query_keys=np.asarray(query_names)); os.replace(tmp, DATA)
    fold_meta: dict[str, Any] = {}
    for f, fd in folds.items():
        fit = fd["fit_groups"]; val = fd["validation_groups"]
        fit_v = sorted({videos[g] for g in fit}); val_v = sorted({videos[g] for g in val}); fit_c = sorted({categories[g] for g in fit}); val_c = sorted({categories[g] for g in val})
        fold_meta[f] = {**fd, "fit_videos": fit_v, "validation_videos": val_v, "fit_categories": fit_c, "validation_categories": val_c, "video_overlap": sorted(set(fit_v) & set(val_v)), "category_overlap": sorted(set(fit_c) & set(val_c)), "video_disjoint": not (set(fit_v) & set(val_v)), "category_disjoint": not (set(fit_c) & set(val_c))}
    manifest = {"schema_version": "trackocd.phase84.b84sq.query_conditioned_manifest.v1", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "data": str(DATA.resolve()), "data_sha256": sha(DATA), "native": str(NATIVE_PATH.resolve()), "native_sha256": sha(NATIVE_PATH), "native_features": str(FEATURE_PATH.resolve()), "native_features_sha256": sha(FEATURE_PATH), "source_cache": str(SOURCE_CACHE.resolve()), "source_cache_sha256": sha(SOURCE_CACHE), "episode_dir": str(EPISODE_DIR.resolve()), "public_csv": str(PUBLIC_CSV.resolve()), "groups": len(targets), "candidate_rows": len(features), "feature_dim": 19, "feature_names": ["frozen_native_candidate_descriptors", "source_mean_cosine", "source_proto_max_cosine", "source_proto_mean_cosine", "source_proto_min_cosine"], "folds": fold_meta, "event_videos_excluded": sorted(v for v in blocked_videos if v >= 0), "source_per_track_cap": {"positive": 2, "hard_defer": 2}, "target_contract": "positive iff candidate IoU>=0.5 to a TRAIN GT box whose category equals source category; otherwise explicit DEFER", "labels_used_only_for_train_targets": True, "model_input_forbidden": ["category", "gt_bbox", "gt_iou", "assigned", "physical_id", "semantic_id", "future", "text", "event_key", "StateMemory", "controller_action"], "mapped_public_source_rows": mapped_rows, "source_tracks_with_native_mapping": len(mapped_by_track), "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(MANIFEST, manifest); atomic_json(OUT / "status.json", {"phase": "Phase84", "route": "B84S_Q_MANIFEST", "status": "COMPLETE", "manifest": str(MANIFEST.resolve()), "public_dev_q1_sealed_accessed": False}); atomic_json(OUT / "completion/b84sq_manifest.done", {"status": "DONE", "manifest": str(MANIFEST.resolve()), "data": str(DATA.resolve())}); print(json.dumps({"groups": len(targets), "candidate_rows": len(features), "feature_dim": 19, "folds": {f: {k: len(v) for k, v in d.items() if k.endswith("groups")} for f, d in fold_meta.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
