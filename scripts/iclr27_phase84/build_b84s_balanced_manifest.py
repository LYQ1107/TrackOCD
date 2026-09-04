#!/usr/bin/env python3
"""Build a balanced, source-conditioned B84S-Q training manifest.

This is a data-contract repair of B84S, not a new model: groups are formed
from the Phase30 query/support pairs, capped per source, and assigned to
deterministic category-balanced validation folds.  All native candidate rows
remain in each listwise action space.
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
from collections import Counter, defaultdict
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
EPISODE_DIR = ROOT / "outputs/iclr27_phase30/manifests"
PUBLIC_CSV = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
SOURCE_CACHE = BASE / "manifests/source_track_native_vectors.npz"
OUT = ROOT / "outputs/iclr27_phase84"
DATA = BASE / "manifests/b84sq_balanced_v2_features.npz"
MANIFEST = OUT / "manifests/b84sq_balanced_v2_manifest.json"


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
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box(value: Any) -> list[float] | None:
    if value is None or value == "": return None
    try:
        x = [float(v) for v in (value if isinstance(value, (list, tuple)) else ast.literal_eval(str(value)))]
        return x if len(x) == 4 else None
    except Exception: return None


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32); return v / max(float(np.linalg.norm(v)), 1e-8)


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1); aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1]); bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-8)


def main() -> None:
    table = load_frozen_tracks()
    public = list(csv.DictReader(PUBLIC_CSV.open(newline="", encoding="utf-8")))
    native = [json.loads(line) for line in NATIVE_PATH.open(encoding="utf-8") if line.strip()]
    native_feat = np.asarray(np.load(FEATURE_PATH, allow_pickle=False)["features"], dtype=np.float32); native_feat /= np.maximum(np.linalg.norm(native_feat, axis=1, keepdims=True), 1e-8)
    if len(native) != len(native_feat): raise RuntimeError("native/features row mismatch")
    b4 = np.load(B4_PATH, allow_pickle=False); descriptors = np.zeros((len(native), 15), dtype=np.float32); descriptors[b4["flat_indices"].astype(np.int64)] = b4["features"].astype(np.float32)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(native):
        if box(row.get("bbox_xyxy")) is not None: groups[(int(row["video_id"]), int(row.get("image_id", -1)))].append(i)
    for key in groups: groups[key].sort(key=lambda i: (int(native[i].get("candidate_rank") or 0), int(native[i].get("physical_track_id", -1)), i))
    public_by_track: dict[str, list[dict[str, str]]] = defaultdict(list); gt_by_image: dict[tuple[int, int], list[tuple[list[float], int]]] = defaultdict(list)
    for row in public:
        public_by_track[track_key(row)].append(row); gb = box(row.get("gt_bbox_xyxy"))
        if gb is not None: gt_by_image[(int(row["video_id"]), int(row["image_id"]))].append((gb, int(float(row.get("gt_category_id_common", -1) or -1))))
    for k in public_by_track: public_by_track[k].sort(key=order_key)
    mapped_by_track: dict[str, list[int]] = defaultdict(list)
    for k, rows in public_by_track.items():
        for row in rows:
            pb = box(row.get("bbox_xyxy")); inds = groups.get((int(row["video_id"]), int(row["image_id"])), [])
            if pb is None or not inds: continue
            j = max(inds, key=lambda q: (iou(pb, box(native[q].get("bbox_xyxy"))), float(native[q].get("base_score", 0.0) or 0.0), -int(native[q].get("candidate_rank") or 0), -q))
            if iou(pb, box(native[j].get("bbox_xyxy"))) >= 0.5: mapped_by_track[k].append(int(j))
    source_z = np.load(SOURCE_CACHE, allow_pickle=False); source_keys = [str(x) for x in source_z["keys"].tolist()]; source_idx = {k: i for i, k in enumerate(source_keys)}; source_v = source_z["vectors"].astype(np.float32); source_p = source_z["prototypes"].astype(np.float32)
    blocked: set[int] = set()
    for line in OBS_PATH.open(encoding="utf-8"):
        if line.strip():
            e = json.loads(line); blocked.update({int(e.get("source_video", -1)), int(e.get("target_video", -1))})
    raw: list[dict[str, Any]] = []
    for fi in range(4):
        ep = json.loads((EPISODE_DIR / f"episode_manifest_f{fi}.json").read_text(encoding="utf-8"))
        for rec in ep["records"]:
            kind = str(rec.get("kind")); split = str(rec.get("split")); q = str(rec.get("query_track_key"))
            if split not in {"fit", "val"} or kind not in {"multi_positive_cross_video", "null_no_match_hard_negative"}: continue
            qm = table.metadata.get(q)
            if qm is None: continue
            qv, qcat = int(qm["video"]), int(qm["category"])
            if qv in blocked or not public_by_track.get(q): continue
            target_row = public_by_track[q][min(15, len(public_by_track[q])-1)]; image_key = (qv, int(target_row["image_id"]))
            if not groups.get(image_key): continue
            for s in sorted({str(x) for x in rec.get("support_track_keys", [])}):
                sm = table.metadata.get(s)
                if sm is None or s not in source_idx or not mapped_by_track.get(s): continue
                svideo = int(sm["video"])
                if svideo == qv or svideo in blocked: continue
                raw.append({"kind": kind, "source_key": s, "query_key": q, "source_category": int(sm["category"]), "query_category": qcat, "source_video": svideo, "target_video": qv, "target_image": image_key[1], "target_row_key": str(target_row.get("row_key", "")), "episode_id": str(rec.get("episode_id")), "orig_fold": fi})
    raw.sort(key=lambda x: (x["source_key"], x["kind"], x["episode_id"], x["query_key"]))
    selected: list[dict[str, Any]] = []; counts: Counter[tuple[str, str, int]] = Counter()
    for c in raw:
        typ = "positive" if c["kind"] == "multi_positive_cross_video" else "hard_defer"; key = (c["source_key"], typ, int(c.get("orig_fold", 0)))
        if counts[key] >= 2: continue
        counts[key] += 1; selected.append(c)
    if not selected: raise RuntimeError("no legal source/query groups after filtering")
    # Greedily partition category units by incident group count, then derive
    # each fold's video/category-disjoint fit set.  No event result is used.
    cat_weight: Counter[int] = Counter()
    for c in selected: cat_weight.update({int(c["query_category"]), int(c["source_category"])})
    def partition(k: int) -> list[set[int]]:
        bins = [set() for _ in range(k)]; weights = [0]*k
        for cat, wt in sorted(cat_weight.items(), key=lambda x: (-x[1], x[0])):
            j = min(range(k), key=lambda z: (weights[z], z)); bins[j].add(cat); weights[j] += wt
        return bins
    # Prefer four folds, but objectively fall back to three if a fold would
    # have fewer than 100 validation groups or 100 fit groups.
    chosen_k = 4; chosen_bins = partition(chosen_k)
    def fold_sets(k: int, bins: list[set[int]], strict_source: bool = True) -> list[dict[str, list[int]]]:
        out = []
        for fi in range(k):
            held = bins[fi]; val = [i for i,c in enumerate(selected) if int(c["query_category"]) in held]
            held_c = {int(selected[i]["query_category"]) for i in val}
            held_v = {int(selected[i]["target_video"]) for i in val}
            # Query categories and target videos define the held-out task.  A
            # source is a legal prior support and may come from another video;
            # its category is metadata for TRAIN target construction only.
            if strict_source:
                held_c = held_c | {int(selected[i]["source_category"]) for i in val}
                held_v = held_v | {int(selected[i]["source_video"]) for i in val}
            fit = [i for i,c in enumerate(selected) if i not in set(val) and int(c["query_category"]) not in held_c and int(c["target_video"]) not in held_v and (not strict_source or (int(c["source_category"]) not in held_c and int(c["source_video"]) not in held_v))]
            out.append({"fit_groups": fit, "validation_groups": val})
        return out
    split_mode = "strict_query_source_category_and_all_video"
    sets = fold_sets(chosen_k, chosen_bins, strict_source=True)
    if min(len(x["validation_groups"]) for x in sets) < 100 or min(len(x["fit_groups"]) for x in sets) < 100:
        split_mode = "query_category_and_target_video_disjoint_prior_support"
        sets = fold_sets(chosen_k, chosen_bins, strict_source=False)
    if min(len(x["validation_groups"]) for x in sets) < 100 or min(len(x["fit_groups"]) for x in sets) < 100:
        chosen_k = 3; chosen_bins = partition(chosen_k); split_mode = "query_category_and_target_video_disjoint_prior_support_3fold"
        sets = fold_sets(chosen_k, chosen_bins, strict_source=False)
    # Materialize each source-conditioned native action group once.
    feat_rows: list[list[float]] = []; offsets = [0]; targets: list[int] = []; videos: list[int] = []; categories: list[int] = []; source_names: list[str] = []; query_names: list[str] = []
    for c in selected:
        inds = groups[(int(c["target_video"]), int(c["target_image"]))]; base = descriptors[np.asarray(inds, dtype=np.int64)]; si = source_idx[c["source_key"]]; sv = norm(source_v[4, si]); ps = [norm(x) for x in source_p[:, si] if float(np.linalg.norm(x)) > 1e-8] or [sv]; z = native_feat[np.asarray(inds, dtype=np.int64)]; sims = z @ sv; pm = np.stack([z @ p for p in ps], axis=1); extra = np.stack([sims, pm.max(1), pm.mean(1), pm.min(1)], axis=1).astype(np.float32); x = np.concatenate([base, extra], axis=1); target = len(inds)
        if c["kind"] == "multi_positive_cross_video":
            vals = []
            for j, ni in enumerate(inds):
                cb = box(native[ni].get("bbox_xyxy")); best_i, best_cat = max(((iou(cb, gb), cat) for gb, cat in gt_by_image.get((int(c["target_video"]), int(c["target_image"])), [])), default=(0.0, -1))
                if best_i >= 0.5 and best_cat == int(c["source_category"]): vals.append((j, best_i, float(native[ni].get("base_score", 0.0) or 0.0), -int(native[ni].get("candidate_rank") or 0)))
            if vals: target = int(max(vals, key=lambda y: (y[1], y[2], y[3]))[0])
        feat_rows.extend(x.tolist()); offsets.append(len(feat_rows)); targets.append(target); videos.append(int(c["target_video"])); categories.append(int(c["query_category"])); source_names.append(c["source_key"]); query_names.append(c["query_key"])
    DATA.parent.mkdir(parents=True, exist_ok=True); tmp = DATA.with_name("." + DATA.name + ".tmp.npz"); np.savez(tmp, features=np.asarray(feat_rows, np.float32), offsets=np.asarray(offsets, np.int64), targets=np.asarray(targets, np.int64), videos=np.asarray(videos, np.int64), categories=np.asarray(categories, np.int64), source_keys=np.asarray(source_names), query_keys=np.asarray(query_names)); os.replace(tmp, DATA)
    fold_meta: dict[str, Any] = {}
    for fi, fd in enumerate(sets):
        fit, val = fd["fit_groups"], fd["validation_groups"]; fcat = sorted({categories[i] for i in fit}); vcat = sorted({categories[i] for i in val}); fvid = sorted({videos[i] for i in fit}); vvid = sorted({videos[i] for i in val}); fold_meta[str(fi)] = {"fit_groups": fit, "validation_groups": val, "fit_categories": fcat, "validation_categories": vcat, "fit_videos": fvid, "validation_videos": vvid, "category_overlap": sorted(set(fcat)&set(vcat)), "video_overlap": sorted(set(fvid)&set(vvid)), "query_category_disjoint": not (set(fcat)&set(vcat)), "target_video_disjoint": not (set(fvid)&set(vvid)), "source_support_may_cross_fold": split_mode != "strict_query_source_category_and_all_video"}
    manifest = {"schema_version": "trackocd.phase84.b84sq.balanced_manifest.v2", "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "data": str(DATA.resolve()), "data_sha256": sha(DATA), "native": str(NATIVE_PATH.resolve()), "native_sha256": sha(NATIVE_PATH), "native_features": str(FEATURE_PATH.resolve()), "native_features_sha256": sha(FEATURE_PATH), "source_cache": str(SOURCE_CACHE.resolve()), "source_cache_sha256": sha(SOURCE_CACHE), "episode_dir": str(EPISODE_DIR.resolve()), "public_csv": str(PUBLIC_CSV.resolve()), "groups": len(selected), "candidate_rows": len(feat_rows), "feature_dim": 19, "fold_count": chosen_k, "fold_assignment": split_mode, "folds": fold_meta, "raw_legal_pair_records": len(raw), "source_cap": {"positive": 2, "hard_defer": 2, "scope": "original Phase30 provenance fold"}, "event_videos_excluded": sorted(v for v in blocked if v >= 0), "target_contract": "positive iff candidate IoU>=0.5 to TRAIN GT box with source category; null/hard-negative is explicit DEFER", "labels_used_only_for_train_targets": True, "model_input_forbidden": ["category", "gt_bbox", "gt_iou", "assigned", "physical_id", "semantic_id", "future", "text", "event_key", "StateMemory", "controller_action"], "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(MANIFEST, manifest); atomic_json(OUT / "status.json", {"phase": "Phase84", "route": "B84S_Q_BALANCED_MANIFEST", "status": "COMPLETE", "manifest": str(MANIFEST.resolve()), "fold_count": chosen_k, "public_dev_q1_sealed_accessed": False}); atomic_json(OUT / "completion/b84sq_balanced_v2_manifest.done", {"status": "DONE", "manifest": str(MANIFEST.resolve()), "data": str(DATA.resolve())}); print(json.dumps({"groups": len(selected), "raw_records": len(raw), "candidate_rows": len(feat_rows), "fold_count": chosen_k, "folds": {f: {"fit": len(d["fit_groups"]), "val": len(d["validation_groups"]), "query_category_disjoint": d["query_category_disjoint"], "target_video_disjoint": d["target_video_disjoint"]} for f,d in fold_meta.items()}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
