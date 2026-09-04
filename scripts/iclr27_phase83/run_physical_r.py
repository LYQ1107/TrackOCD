#!/usr/bin/env python3
"""Phase83 Branch-A: physical lineage -> frozen raw R diagnostic.

The native Phase82R lineage covers only the event-video stream.  We therefore
report two views: an exact Phase75D validation universe with an explicit raw
fallback for unmapped tracks, and a mapped-only diagnostic whose denominator is
reported (never treated as a formal gate).  No held-event labels are used in
the TRAIN validation scorer; event rows are used only for a separate
post-hoc mapping diagnostic.
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

# Direct script execution (the reproducibility command in the report) does
# not automatically put the repository root on sys.path.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iclr27_phase75d.protocol import load_frozen_tracks, PREFIXES
from src.iclr27_phase75d.retrieval_metrics import score_records, aggregate_fold_metrics

OUT = ROOT / "outputs/iclr27_phase83"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase75b/event_full_sequence_repair2/native_lineage.jsonl")
NATIVE_FEAT = ROOT / "outputs/iclr27_phase82r/features/native_dinov2_corrected_r1.npz"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
PUBLIC_ROWS = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
POS_EVENTS = ROOT / "outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]); bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-8)


def norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32); return x / max(float(np.linalg.norm(x)), 1e-8)


def load_native() -> tuple[list[dict[str, Any]], dict[tuple[int, int], list[int]], np.ndarray]:
    rows: list[dict[str, Any]] = []; by_image: dict[tuple[int, int], list[int]] = defaultdict(list)
    with NATIVE.open(encoding="utf-8") as f:
        for j, line in enumerate(f):
            d = json.loads(line); rows.append(d)
            if d.get("bbox_xyxy") is not None: by_image[(int(d["video_id"]), int(d["image_id"]))].append(j)
    feat = np.load(NATIVE_FEAT, allow_pickle=False)["features"].astype(np.float32)
    if len(rows) != feat.shape[0]: raise RuntimeError(f"native lineage/features mismatch {len(rows)} vs {feat.shape[0]}")
    return rows, by_image, feat


def public_native_mapping(public: list[dict[str, str]], native: list[dict[str, Any]], by_image: dict[tuple[int, int], list[int]]) -> tuple[np.ndarray, np.ndarray]:
    best = np.zeros(len(public), dtype=np.float32); best_idx = np.full(len(public), -1, dtype=np.int64)
    for i, r in enumerate(public):
        try: pb = json.loads(r["bbox_xyxy"])
        except Exception: continue
        cand = by_image.get((int(r["video_id"]), int(r["image_id"])), [])
        vals = [(iou(pb, native[j].get("bbox_xyxy")), j) for j in cand]
        if vals:
            score, j = max(vals); best[i] = score; best_idx[i] = j
    return best, best_idx


def temporal_vectors(public: list[dict[str, str]], features: np.ndarray, best: np.ndarray, best_idx: np.ndarray) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    tracks: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(public): tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(i)
    matched: dict[str, list[int]] = {}; info: dict[str, dict[str, Any]] = {}
    for key, inds in tracks.items():
        inds = sorted(inds, key=lambda i: (int(public[i].get("event_rank", 0)), int(public[i].get("frame_id", 0)), int(public[i].get("proposal_local_id", 0))))
        good = [i for i in inds if best_idx[i] >= 0 and best[i] >= 0.5]
        if good: matched[key] = good
        info[key] = {"rows": len(inds), "matched_rows_ge_iou_0.5": len(good), "mapping_fraction": len(good) / max(len(inds), 1), "video": int(public[inds[-1]]["video_id"])}
    return matched, info


def build_records(table, vectors: dict[str, np.ndarray], fold: int, prefix: int, *, fallback_raw: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = EPISODES / f"episode_manifest_f{fold}.json"; manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    keys = sorted({str(r["query_track_key"]) for r in manifest["records"] if r.get("split") == "val" and str(r.get("query_track_key")) in table.metadata})
    available = [k for k in keys if k in vectors or fallback_raw]
    vals = {k: vectors.get(k, table.raw_vector(k, prefix)) for k in available}
    all_idx = np.arange(len(available), dtype=np.int64); vids = np.asarray([table.metadata[k]["video"] for k in available]); cats = np.asarray([table.metadata[k]["category"] for k in available])
    recs: list[dict[str, Any]] = []
    for i, q in enumerate(available):
        ci = all_idx[(all_idx != i) & (vids != vids[i])]; candidates = [available[int(j)] for j in ci]
        pos = [available[int(j)] for j in ci if cats[int(j)] == cats[i]]; neg = [available[int(j)] for j in ci if cats[int(j)] != cats[i]]
        qv = vals[q]; scores = [float(qv @ vals[c]) for c in candidates]; raw = [float(table.raw_vector(q, prefix) @ table.raw_vector(c, prefix)) for c in candidates]
        recs.append({"query_key": q, "category": int(cats[i]), "video": int(vids[i]), "candidates": candidates, "positives": pos, "negatives": neg, "scores": scores, "raw_scores": raw})
    return recs, {"fold": fold, "prefix": prefix, "manifest": str(manifest_path.resolve()), "manifest_sha256": sha(manifest_path), "keys_total": len(keys), "keys_evaluable": len(available), "candidate_universe": "all validation tracks except self and same video", "raw_fallback_for_unmapped": bool(fallback_raw)}


def compact(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m[k] for k in ("queries", "r1", "r5", "map", "raw_r1", "raw_r5", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "category_macro_r1", "video_macro_r1", "unsafe_flip_count", "unsafe_flip_micro_rate", "top1_change_count", "top1_change_rate")}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-id", default="phase83-physical-r-temporal-20260904"); args = parser.parse_args()
    table = load_frozen_tracks(); public = list(csv.DictReader(PUBLIC_ROWS.open(newline="", encoding="utf-8"))); native, by_image, native_feat = load_native()
    best, best_idx = public_native_mapping(public, native, by_image); matched, map_info = temporal_vectors(public, native_feat, best, best_idx)
    # Build causal vectors from matched current/past rows only; no suffix is
    # read.  Prefix is clipped after matching and therefore explicitly audited.
    temporal_by_prefix: dict[int, dict[str, np.ndarray]] = {}
    for p in PREFIXES:
        vv: dict[str, np.ndarray] = {}
        for key, inds in matched.items():
            use = inds[:p]
            if use: vv[key] = norm(np.mean([native_feat[best_idx[i]] for i in use], axis=0))
        temporal_by_prefix[p] = vv
    sections: dict[str, Any] = {}
    for mode, fallback in (("exact_mixed", True), ("mapped_subset", False)):
        folds: list[dict[str, Any]] = []; prefix_rows: list[dict[str, Any]] = []
        for fold in range(4):
            for p in PREFIXES:
                recs, inv = build_records(table, temporal_by_prefix[p], fold, p, fallback_raw=fallback)
                mm = score_records(recs); c = compact(mm)
                folds.append({"fold": fold, "prefix": p, "metrics": c, "inventory": inv})
                prefix_rows.append({"fold": fold, "prefix": p, "metrics": c, "keys_total": inv["keys_total"], "keys_evaluable": inv["keys_evaluable"]})
        agg: dict[str, Any] = {}
        for p in PREFIXES:
            fs = [x["metrics"] for x in folds if x["prefix"] == p]
            agg[str(p)] = aggregate_fold_metrics(fs)
        p16 = [x for x in folds if x["prefix"] == 16]
        fold_direction = [float(x["metrics"]["r1"]) >= float(x["metrics"]["raw_r1"]) and float(x["metrics"]["map"]) >= float(x["metrics"]["raw_map"]) for x in p16]
        sections[mode] = {"folds": folds, "prefix": agg, "prefix_rows": prefix_rows, "gate_diagnostic": {"p16_fold_both_non_decrease": fold_direction, "folds_non_decrease": int(sum(fold_direction)), "unsafe_flip_count": int(sum(x["metrics"]["unsafe_flip_count"] for x in p16)), "formal_gate_eligible": mode == "exact_mixed" and sum(fold_direction) >= 3 and sum(x["metrics"]["unsafe_flip_count"] for x in p16) == 0}}
    # Event 76 diagnostic: mapped physical vectors are compared pairwise for
    # known positive source/target tracks.  GT is read only from the frozen
    # event manifest for post-hoc reporting, never for vector construction.
    events = [json.loads(x) for x in POS_EVENTS.read_text(encoding="utf-8").splitlines() if x.strip()]
    event_rows = []
    for e in events:
        sk = str(e["source_tracklet_keys"][0]); tk = str(e["target_tracklet_key"]); sr = temporal_by_prefix[16].get(sk); tr = temporal_by_prefix[16].get(tk); raw_s = table.raw_vector(sk, 16) if sk in table.sequences else None; raw_t = table.raw_vector(tk, 16) if tk in table.sequences else None
        event_rows.append({"event_key": e["event_key"], "fold": int(e["fold"]), "category": e.get("category_gt_denominator_only"), "source_track_mapped": sr is not None, "target_track_mapped": tr is not None, "both_mapped": sr is not None and tr is not None, "temporal_cosine": None if sr is None or tr is None else float(sr @ tr), "raw_cosine": None if raw_s is None or raw_t is None else float(raw_s @ raw_t)})
    event_both = [x for x in event_rows if x["both_mapped"]]
    event_diag = {"denominator": 76, "both_mapped": len(event_both), "source_mapped": sum(x["source_track_mapped"] for x in event_rows), "target_mapped": sum(x["target_track_mapped"] for x in event_rows), "mean_temporal_cosine": float(np.mean([x["temporal_cosine"] for x in event_both])) if event_both else None, "mean_raw_cosine": float(np.mean([x["raw_cosine"] for x in event_both])) if event_both else None, "events": event_rows, "labels_used_for": "post-hoc mapping diagnostic only"}
    mapping = {"schema_version": "trackocd.phase83.native_mapping.v1", "native_path": str(NATIVE.resolve()), "native_sha256": sha(NATIVE), "native_feature_path": str(NATIVE_FEAT.resolve()), "native_feature_sha256": sha(NATIVE_FEAT), "public_csv_sha256": sha(PUBLIC_ROWS), "public_rows": len(public), "native_rows": len(native), "row_best_iou_ge_0.5": int(np.sum(best >= 0.5)), "row_best_iou_quantiles": [float(x) for x in np.quantile(best, [0, .1, .5, .9, 1])], "mapped_track_count": len(matched), "track_count": len(map_info), "mapped_track_fraction": len(matched) / max(len(map_info), 1), "mapping_rule": "same (video_id,image_id), max bbox IoU >= 0.5; temporal mean uses only matched current/past rows", "event_video_stream_only": True}
    atomic_json(OUT / "audit/native_mapping.json", mapping); atomic_json(OUT / "audit/event_physical_r_diagnostic.json", event_diag)
    atomic_json(OUT / "metrics/physical_r_temporal.json", {"schema_version": "trackocd.phase83.physical_r.v1", "phase": "Phase83 Branch A", "run_id": args.run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "sections": sections, "mapping": mapping, "event_diagnostic": {k: v for k, v in event_diag.items() if k != "events"}, "raw_reference": str((ROOT / "outputs/iclr27_phase75d/metrics/global_r.json").resolve()), "public_dev_q1_sealed_accessed": False, "future_rows_or_tracks": False, "ids_as_model_input": False, "held_events_used_for_model_selection": False})
    atomic_json(OUT / "audit/event_physical_r_diagnostic_events.json", event_diag)
    p16 = sections["exact_mixed"]["prefix"]["16"]
    status = "R83_DIAGNOSTIC_NO_SAFE_IMPROVEMENT" if not sections["exact_mixed"]["gate_diagnostic"]["formal_gate_eligible"] else "R83_DIAGNOSTIC_SAFE_NONDECREASE"
    atomic_json(OUT / "status.json", {"phase": "Phase83", "status": status, "run_id": args.run_id, "next_action": "run O-support TRAIN-only router audit/training" if status != "R83_GATE_PASS" else "register one downstream R/C route", "physical_r": {"exact_mixed_p16": p16, "mapped_subset_p16": sections["mapped_subset"]["prefix"]["16"], "event_diagnostic": event_diag}, "public_dev_q1_sealed_accessed": False, "resource_event": "single-process CPU"})
    atomic_json(OUT / "completion/physical_r.done", {"status": status, "metrics": str(OUT / "metrics/physical_r_temporal.json"), "mapping": str(OUT / "audit/native_mapping.json")})
    print(json.dumps({"status": status, "exact_mixed_p16": {k: p16[k] for k in ("queries", "r1", "raw_r1", "map", "raw_map", "hard_negative_gap", "raw_hard_negative_gap", "unsafe_flip_count")}, "mapped_tracks": mapping["mapped_track_count"], "event_both_mapped": event_diag["both_mapped"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
