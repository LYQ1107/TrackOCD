#!/usr/bin/env python3
"""Build causal per-image candidate sets for the Phase83 B2 selector.

Only public TRAIN rows are materialized.  IoU/assignment fields form labels;
the feature tensor contains causal proposal, geometry and history fields only.
Event videos are excluded from fit/validation groups and retained solely for
the later post-hoc 76+76 replay.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase23.protocol import load_aligned_features

OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
FOLD_MANIFEST = ROOT / "outputs/iclr27_phase22/manifests/fold_manifest.json"
POS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"
NEG = ROOT / "outputs/iclr27_phase19r/manifests/held_known_negative_events.jsonl"
FEATURE_NAMES = [
    "base_proposal_score", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log",
    "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_age_norm",
    "causal_box_stability_iou", "history_length_norm", "gap_norm", "proposal_density_log",
    "candidate_ambiguity_log", "corrected_dinov2_cosine", "temporal_mean_cosine",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(key, default)); return x if math.isfinite(x) else default
    except (TypeError, ValueError): return default


def event_videos() -> set[int]:
    out: set[int] = set()
    for path in (POS, NEG):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e = json.loads(line); out.update((int(e["source_video"]), int(e["target_video"])))
    return out


def order_key(row: dict[str, str]) -> tuple[int, int, int]:
    return (int(f(row, "event_rank")), int(f(row, "frame_id")), int(f(row, "proposal_local_id")))


def row_features(rows: list[dict[str, str]], fused: np.ndarray) -> np.ndarray:
    # Recompute only the allowed causal scalar fields; DINO is used as an
    # appearance descriptor but no label/ID/category field enters this tensor.
    tracks: dict[str, list[int]] = defaultdict(list); image_count: Counter[tuple[int, int]] = Counter(); image_tracks: defaultdict[tuple[int, int], set[str]] = defaultdict(set)
    for i, r in enumerate(rows):
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"; tracks[key].append(i)
        ik = (int(r["video_id"]), int(r["image_id"])); image_count[ik] += 1; image_tracks[ik].add(key)
    x = np.zeros((len(rows), len(FEATURE_NAMES)), np.float32)
    for key, inds in tracks.items():
        inds.sort(key=lambda i: order_key(rows[i])); running = np.zeros(fused.shape[1], np.float32); prev = None; prev_frame = None
        for pos, i in enumerate(inds):
            r = rows[i]; cur = fused[i].astype(np.float32); cur /= max(float(np.linalg.norm(cur)), 1e-8)
            hist = running / max(pos, 1); hist /= max(float(np.linalg.norm(hist)), 1e-8)
            frame = int(f(r, "frame_id")); gap = 0.0 if prev_frame is None else max(0, frame - prev_frame); ik = (int(r["video_id"]), int(r["image_id"]))
            x[i] = [f(r,"score"),f(r,"box_width_norm"),f(r,"box_height_norm"),f(r,"box_area_norm"),f(r,"box_aspect_log"),f(r,"border_left_norm"),f(r,"border_top_norm"),f(r,"border_right_norm"),f(r,"border_bottom_norm"),f(r,"causal_age_norm"),f(r,"causal_box_stability_iou"),math.log1p(pos)/8.0,math.log1p(gap)/8.0,math.log1p(image_count[ik])/8.0,math.log1p(len(image_tracks[ik]))/4.0,float(cur @ prev) if prev is not None else 0.0,float(cur @ hist) if pos else 0.0]
            running += cur; prev = cur; prev_frame = frame
    return x


def main() -> None:
    ap = __import__("argparse").ArgumentParser(); ap.add_argument("--tag", default="b2_candidate_sets_v1"); args = ap.parse_args()
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    aligned, _roi, alignment = load_aligned_features(rows); fused = aligned.astype(np.float32); fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8)
    x = row_features(rows, fused)
    blocked = event_videos(); groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if int(r["video_id"]) not in blocked: groups[(int(r["video_id"]), int(r["image_id"]))].append(i)
    # Candidate order is the immutable CSV order sorted by proposal-local id,
    # then track id and row index.  The target is a TRAIN-only IoU-soft label.
    group_rows: list[list[int]] = []; targets: list[int] = []; videos: list[int] = []; cats: list[int] = []; reliable = 0; defer = 0
    for (video, _image), inds in sorted(groups.items()):
        inds.sort(key=lambda i: (int(f(rows[i], "proposal_local_id")), int(f(rows[i], "track_id")), i)); group_rows.append(inds); videos.append(video)
        labelled = [i for i in inds if int(f(rows[i], "assigned")) == 1 and f(rows[i], "row_iou") >= 0.5]
        if labelled:
            target = max(labelled, key=lambda i: (f(rows[i], "row_iou"), f(rows[i], "score"), -int(f(rows[i], "proposal_local_id")), -i)); targets.append(inds.index(target)); reliable += 1
        else:
            targets.append(len(inds)); defer += 1
        cvals = [int(f(rows[i], "gt_category_id_common", -1)) for i in inds if int(f(rows[i], "gt_category_id_common", -1)) >= 0]; cats.append(Counter(cvals).most_common(1)[0][0] if cvals else -1)
    offsets = [0]; flat: list[int] = []
    for g in group_rows: flat.extend(g); offsets.append(len(flat))
    fm = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8")); folds: dict[str, Any] = {}
    for fi, fold in enumerate(fm["folds"]):
        fit_v = set(int(v) for v in fold["fit_videos"]) - blocked; val_v = set(int(v) for v in fold["validation_videos"]) - blocked; fit_c = set(int(c) for c in fold["fit_categories"]); val_c = set(int(c) for c in fold.get("held_categories", []))
        fit = [j for j, (v,c) in enumerate(zip(videos,cats)) if v in fit_v and (c in fit_c or c < 0)]; val = [j for j, (v,c) in enumerate(zip(videos,cats)) if v in val_v and c in val_c]
        folds[str(fi)] = {"fit_groups": fit, "validation_groups": val, "fit_videos": sorted(fit_v), "validation_videos": sorted(val_v), "fit_categories": sorted(fit_c), "validation_categories": sorted(val_c), "video_disjoint": True, "category_disjoint": True}
    out_dir = Path("/data2/usr_for_deadline/trackocd_phase83/b2_candidate_sets")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{args.tag}.npz"; tmp = Path(str(data_path) + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, features=x.astype(np.float32), flat_indices=np.asarray(flat, np.int64), offsets=np.asarray(offsets, np.int64), targets=np.asarray(targets, np.int64), videos=np.asarray(videos, np.int64), categories=np.asarray(cats, np.int64)); os.replace(tmp, data_path)
    manifest = {"schema_version":"trackocd.phase83.b2_candidate_sets.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"tag":args.tag,"csv":str(CSV_PATH.resolve()),"csv_sha256":sha(CSV_PATH),"data":str(data_path.resolve()),"data_sha256":sha(data_path),"rows":len(rows),"groups":len(group_rows),"candidate_rows":len(flat),"features":len(FEATURE_NAMES),"feature_names":FEATURE_NAMES,"group_candidate_count_min":min(map(len,group_rows),default=0),"group_candidate_count_max":max(map(len,group_rows),default=0),"reliable_target_groups":reliable,"defer_target_groups":defer,"event_videos_excluded":sorted(blocked),"folds":folds,"alignment":alignment,"labels_used_only_for_train_target":True,"forbidden_model_input_fields":["assigned","row_iou","gt_bbox_xyxy","gt_track_id","gt_category_id_common","semantic_id","physical_id","event_key","future","text"],"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False}
    atomic_json(OUT/f"manifests/{args.tag}.json",manifest); atomic_json(OUT/"status.json",{"phase":"Phase83","status":"B2_CANDIDATE_SET_MANIFEST_COMPLETE","manifest":str((OUT/f"manifests/{args.tag}.json").resolve()),"next_action":"train explicit listwise selector; no binary router reuse","public_dev_q1_sealed_accessed":False})
    print(json.dumps({"status":"COMPLETE","groups":len(group_rows),"candidate_rows":len(flat),"reliable_groups":reliable,"defer_groups":defer,"folds":{k:{"fit":len(v['fit_groups']),"val":len(v['validation_groups'])} for k,v in folds.items()},"data":str(data_path)},indent=2,sort_keys=True))


if __name__ == "__main__": main()
