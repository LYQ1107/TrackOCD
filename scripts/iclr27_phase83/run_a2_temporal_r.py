#!/usr/bin/env python3
"""Evaluate temporal appearance on the complete A2 Q0 physical lineage.

Public Phase30/75D track keys are aligned to the independently exported Q0
physical stream by same-image box IoU.  No feature is filled from the public
raw stream when a mapping is absent: missing tracks remain explicit in the
coverage ledger and are excluded only from the mapped-subset diagnostic.
"""
from __future__ import annotations

import argparse
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
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key
from src.iclr27_phase75d.retrieval_metrics import aggregate_fold_metrics, score_records

OUT = ROOT / "outputs/iclr27_phase83"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
PUBLIC = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
EVENTS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box_iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def load_native() -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[int]], np.ndarray]:
    rows: list[dict[str, Any]] = []
    by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    with NATIVE.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            row = json.loads(line); rows.append(row)
            if row.get("bbox_xyxy") is not None:
                by_image[(int(row["video_id"]), int(row["image_id"]))].append(i)
    feat = np.load(FEATURES, allow_pickle=False)["features"].astype(np.float32)
    if len(rows) != feat.shape[0] or feat.shape[1] != 768:
        raise RuntimeError(f"A2 native/features mismatch rows={len(rows)} feat={feat.shape}")
    return rows, by_image, feat


def map_public(public: list[dict[str, str]], native: list[dict[str, Any]], by_image: dict[tuple[int, int], list[int]]) -> tuple[np.ndarray, np.ndarray]:
    best = np.zeros(len(public), dtype=np.float32)
    best_idx = np.full(len(public), -1, dtype=np.int64)
    for i, row in enumerate(public):
        try: pb = [float(x) for x in json.loads(row["bbox_xyxy"])]
        except Exception: continue
        candidates = by_image.get((int(row["video_id"]), int(row["image_id"])), [])
        if candidates:
            score, idx = max((box_iou(pb, native[j].get("bbox_xyxy")), j) for j in candidates)
            best[i], best_idx[i] = float(score), int(idx)
    return best, best_idx


def temporal_tracks(public: list[dict[str, str]], native: list[dict[str, Any]], feat: np.ndarray, best: np.ndarray, best_idx: np.ndarray) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, dict[str, Any]], dict[int, dict[str, list[int]]]]:
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(public): tracks[track_key(row)].append(i)
    matched: dict[str, list[int]] = {}
    info: dict[str, dict[str, Any]] = {}
    for key, inds in tracks.items():
        inds = sorted(inds, key=lambda i: order_key(public[i]))
        good = [i for i in inds if best_idx[i] >= 0 and best[i] >= 0.5]
        if good: matched[key] = good
        info[key] = {
            "rows": len(inds), "matched_rows_ge_iou_0.5": len(good),
            "mapping_fraction": len(good) / max(1, len(inds)),
            "video": int(public[inds[-1]]["video_id"]),
            "native_track_ids": sorted({int(native[best_idx[i]]["physical_track_id"]) for i in good}),
        }
    vectors: dict[int, dict[str, np.ndarray]] = {p: {} for p in PREFIXES}
    for p in PREFIXES:
        for key, inds in matched.items():
            use = inds[:p]
            if use:
                z = np.mean(np.asarray([feat[best_idx[i]] for i in use], dtype=np.float32), axis=0)
                vectors[p][key] = z / max(float(np.linalg.norm(z)), 1e-8)
    return vectors, info, {p: {k: v[:p] for k, v in matched.items()} for p in PREFIXES}


def records(table: Any, vectors: dict[str, np.ndarray], fold: int, prefix: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = EPISODES / f"episode_manifest_f{fold}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = sorted({str(r["query_track_key"]) for r in manifest["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
    available = [k for k in keys if k in vectors]
    missing = [k for k in keys if k not in vectors]
    vids = np.asarray([table.metadata[k]["video"] for k in available], dtype=np.int64)
    cats = np.asarray([table.metadata[k]["category"] for k in available], dtype=np.int64)
    arr = np.asarray([vectors[k] for k in available], dtype=np.float32)
    raw = np.asarray([table.raw_vector(k, prefix) for k in available], dtype=np.float32)
    idx = np.arange(len(available), dtype=np.int64)
    out: list[dict[str, Any]] = []
    for i, q in enumerate(available):
        ci = idx[(idx != i) & (vids != vids[i])]
        cand = [available[int(j)] for j in ci]
        pos = [available[int(j)] for j in ci if cats[int(j)] == cats[i]]
        neg = [available[int(j)] for j in ci if cats[int(j)] != cats[i]]
        out.append({"query_key": q, "category": int(cats[i]), "video": int(vids[i]), "candidates": cand, "positives": pos, "negatives": neg, "scores": [float(arr[i] @ arr[int(j)]) for j in ci], "raw_scores": [float(raw[i] @ raw[int(j)]) for j in ci]})
    return out, {"fold": fold, "prefix": prefix, "keys_total": len(keys), "keys_evaluable": len(available), "missing_keys": missing, "candidate_universe": "all Phase30 validation tracks except self and same video", "raw_fallback_for_unmapped": False, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path)}


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--run-id", default="phase83-a2-temporal-r-20260904"); args = ap.parse_args()
    table = load_frozen_tracks()
    public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8")))
    native, by_image, feat = load_native()
    best, best_idx = map_public(public, native, by_image)
    vectors, mapping_info, prefix_matches = temporal_tracks(public, native, feat, best, best_idx)
    fold_rows: list[dict[str, Any]] = []; prefix_rows: list[dict[str, Any]] = []
    for fold in range(4):
        for p in PREFIXES:
            recs, inv = records(table, vectors[p], fold, p)
            mm = score_records(recs)
            compact = {k: mm[k] for k in ("queries", "r1", "r5", "map", "raw_r1", "raw_r5", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "category_macro_r1", "video_macro_r1", "unsafe_flip_count", "unsafe_flip_micro_rate", "top1_change_count", "top1_change_rate")}
            fold_rows.append({"fold": fold, "prefix": p, "metrics": compact, "inventory": inv})
            prefix_rows.append({"fold": fold, "prefix": p, "metrics": compact, "keys_total": inv["keys_total"], "keys_evaluable": inv["keys_evaluable"], "missing_keys": inv["missing_keys"]})
    aggregate: dict[str, Any] = {}
    for p in PREFIXES:
        fs = [x["metrics"] for x in fold_rows if x["prefix"] == p]
        aggregate[str(p)] = aggregate_fold_metrics(fs)
    p16 = [x for x in fold_rows if x["prefix"] == 16]
    all_complete = all(x["inventory"]["keys_evaluable"] == x["inventory"]["keys_total"] for x in p16)
    direction = [x["metrics"]["r1"] >= x["metrics"]["raw_r1"] and x["metrics"]["map"] >= x["metrics"]["raw_map"] for x in p16]
    event_rows = []
    if EVENTS.is_file():
        for e in (json.loads(x) for x in EVENTS.read_text(encoding="utf-8").splitlines() if x.strip()):
            sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"])
            sv, tv = vectors[16].get(sk), vectors[16].get(tk)
            event_rows.append({"event_key": e["event_key"], "fold": int(e["fold"]), "category": e.get("category_gt_denominator_only"), "source_track_mapped": sv is not None, "target_track_mapped": tv is not None, "both_mapped": sv is not None and tv is not None, "temporal_cosine": None if sv is None or tv is None else float(sv @ tv)})
    mapping = {"schema_version": "trackocd.phase83.a2_native_mapping.v1", "native": str(NATIVE.resolve()), "native_sha256": sha256(NATIVE), "features": str(FEATURES.resolve()), "features_sha256": sha256(FEATURES), "public_csv": str(PUBLIC.resolve()), "public_csv_sha256": sha256(PUBLIC), "public_rows": len(public), "native_rows": len(native), "row_best_iou_ge_0.5": int(np.sum(best >= 0.5)), "row_best_iou_quantiles": [float(x) for x in np.quantile(best, [0, .1, .5, .9, 1])], "mapped_track_count": len(mapping_info), "track_count": len(mapping_info), "track_mapping_fraction": float(sum(1 for x in mapping_info.values() if x["matched_rows_ge_iou_0.5"] > 0) / max(1, len(mapping_info))), "track_full_row_fraction": float(sum(1 for x in mapping_info.values() if x["matched_rows_ge_iou_0.5"] == x["rows"]) / max(1, len(mapping_info))), "mapping_rule": "same (video_id,image_id), max proposal-box IoU >= 0.5; temporal vectors use matched current/past rows only", "event_both_mapped": sum(int(x["both_mapped"]) for x in event_rows), "future_rows_or_tracks": False, "ids_as_model_input": False}
    metrics = {"schema_version": "trackocd.phase83.a2_temporal_r.v1", "phase": "Phase83 A2", "run_id": args.run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "prefix": aggregate, "folds": fold_rows, "prefix_rows": prefix_rows, "gate_diagnostic": {"p16_all_keys_mapped": all_complete, "p16_fold_both_non_decrease": direction, "p16_folds_non_decrease": int(sum(direction)), "p16_unsafe_flip_count": int(sum(x["metrics"]["unsafe_flip_count"] for x in p16)), "headline_raw_fallback": False}, "mapping": mapping, "event_diagnostic": {"denominator": 76, "source_mapped": sum(int(x["source_track_mapped"]) for x in event_rows), "target_mapped": sum(int(x["target_track_mapped"]) for x in event_rows), "both_mapped": sum(int(x["both_mapped"]) for x in event_rows), "events": event_rows}, "public_dev_q1_sealed_accessed": False, "held_events_used_for_model_selection": False, "future_rows_or_tracks": False, "ids_as_model_input": False}
    atomic_json(OUT / "audit/a2_native_mapping.json", mapping)
    atomic_json(OUT / "audit/a2_event_temporal_r.json", metrics["event_diagnostic"])
    atomic_json(OUT / "metrics/a2_temporal_r.json", metrics)
    status = "A2_R_COMPLETE_MAPPED_SUBSET" if all_complete else "A2_R_INCOMPLETE_MAPPING_NO_HEADLINE_GATE"
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": status, "run_id": args.run_id, "next_action": "run B2 contract-level listwise O-support assignment", "a2_temporal_r_p16": aggregate.get("16"), "mapping": mapping, "public_dev_q1_sealed_accessed": False, "headline_raw_fallback": False})
    atomic_json(OUT / "completion/a2_temporal_r.done", {"status": status, "metrics": str(OUT / "metrics/a2_temporal_r.json"), "mapping": str(OUT / "audit/a2_native_mapping.json")})
    print(json.dumps({"status": status, "p16": {k: aggregate["16"].get(k) for k in ("queries", "r1", "raw_r1", "map", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "unsafe_flip_count")}, "mapping": {k: mapping[k] for k in ("row_best_iou_ge_0.5", "mapped_track_count", "track_full_row_fraction", "event_both_mapped")}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
