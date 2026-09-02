#!/usr/bin/env python3
"""Run the single frozen Grounded Correspondence R route.

The route is deliberately parameter-free, following the official ICML 2026
Grounded Correspondence component.  TRAIN-derived disjoint validation episode
manifests provide category/video metadata only for scoring; no 76-event
evaluator rows are read by this script.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, by_track, load_aligned_features, order_key
from src.iclr27_phase75c.grounded_correspondence import GroundedCorrespondence, hungarian_match_score, l2_normalize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase75c"
EPISODE_ROOT = ROOT / "outputs/iclr27_phase30/manifests"
PREFIXES = (1, 2, 4, 8, 16)
R1_DELTA = 0.02
MAP_DELTA = 0.01


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
            json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_data() -> tuple[list[dict[str, str]], dict[str, list[int]], dict[str, dict[str, Any]], np.ndarray, dict[str, Any]]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cls, roi, alignment = load_aligned_features(rows)
    fused = (0.8 * cls.astype(np.float32) + 0.2 * roi.astype(np.float32)).astype(np.float32)
    fused = l2_normalize(fused)
    tracks = by_track(rows)
    metadata: dict[str, dict[str, Any]] = {}
    for key, indices in tracks.items():
        ordered = sorted(indices, key=lambda i: order_key(rows[i]))
        labelled = [rows[i] for i in ordered if rows[i].get("gt_category_id_common") not in {"", "-1", "None", None}]
        if not labelled:
            continue
        category = int(labelled[-1]["gt_category_id_common"])
        if category < 0:
            continue
        metadata[key] = {
            "category": category,
            "video": int(ordered[-1]["video_id"]),
            "indices": ordered,
            "length": len(ordered),
        }
    return rows, tracks, metadata, fused, alignment


def sequence(key: str, metadata: dict[str, dict[str, Any]], features: np.ndarray, prefix: int) -> np.ndarray:
    indices = metadata[key]["indices"][: min(int(prefix), 16)]
    if not indices:
        return np.zeros((0, features.shape[1]), dtype=np.float32)
    return l2_normalize(features[np.asarray(indices, dtype=np.int64)])


def raw_vector(key: str, metadata: dict[str, dict[str, Any]], features: np.ndarray, prefix: int) -> np.ndarray:
    return l2_normalize(sequence(key, metadata, features, prefix).mean(axis=0))


def grounded_vector(key: str, metadata: dict[str, dict[str, Any]], features: np.ndarray, prefix: int, model: GroundedCorrespondence) -> np.ndarray:
    return model.encode(sequence(key, metadata, features, prefix))


def query_metric(keys: list[str], vectors: np.ndarray, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not keys:
        return {"queries": 0, "r1": 0.0, "r5": 0.0, "map": 0.0, "hard_negative_gap": 0.0, "category_macro_r1": 0.0, "video_macro_r1": 0.0, "unsafe_flip": 0}
    sim = vectors @ vectors.T
    videos = np.asarray([metadata[k]["video"] for k in keys], dtype=np.int64)
    categories = np.asarray([metadata[k]["category"] for k in keys], dtype=np.int64)
    all_idx = np.arange(len(keys), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    unsafe = 0
    for i, key in enumerate(keys):
        candidates = all_idx[(all_idx != i) & (videos != videos[i])]
        positives = candidates[categories[candidates] == categories[i]]
        negatives = candidates[categories[candidates] != categories[i]]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        scores = sim[i, candidates]
        order = candidates[np.argsort(scores)[::-1]]
        pos_set = set(int(x) for x in positives)
        hit = np.asarray([int(int(x) in pos_set) for x in order], dtype=np.float32)
        raw_correct = bool(hit[0] > 0)
        r1 = float(hit[:1].max(initial=0.0)); r5 = float(hit[:5].max(initial=0.0))
        cumulative = np.cumsum(hit)
        ap = float(np.sum(cumulative / (np.arange(len(hit)) + 1) * hit) / max(len(positives), 1))
        unsafe += int(raw_correct is False and False)  # populated by paired caller
        rows.append({
            "category": int(categories[i]), "video": int(videos[i]), "r1": r1,
            "r5": r5, "map": ap,
            "hard_negative_gap": float(np.max(scores[np.isin(candidates, positives)]) - np.max(scores[np.isin(candidates, negatives)])),
            "top1_index": int(order[0]),
        })
    by_category: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_video: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row); by_video[row["video"]].append(row)
    mean = lambda field, values: float(np.mean([float(x[field]) for x in values])) if values else 0.0
    return {
        "queries": len(rows), "r1": mean("r1", rows), "r5": mean("r5", rows), "map": mean("map", rows),
        "hard_negative_gap": mean("hard_negative_gap", rows),
        "category_macro_r1": float(np.mean([mean("r1", v) for v in by_category.values()])) if by_category else 0.0,
        "video_macro_r1": float(np.mean([mean("r1", v) for v in by_video.values()])) if by_video else 0.0,
        "category_count": len(by_category), "video_count": len(by_video), "per_query": rows,
    }


def paired_unsafe(keys: list[str], raw: np.ndarray, grounded: np.ndarray, metadata: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count raw-correct -> grounded-wrong flips on the same candidate set."""
    unsafe = 0; compared = 0; changed = 0
    rv, gv = raw @ raw.T, grounded @ grounded.T
    videos = np.asarray([metadata[k]["video"] for k in keys]); cats = np.asarray([metadata[k]["category"] for k in keys]); all_idx = np.arange(len(keys))
    for i in range(len(keys)):
        c = all_idx[(all_idx != i) & (videos != videos[i])]; p = c[cats[c] == cats[i]]; n = c[cats[c] != cats[i]]
        if len(p) == 0 or len(n) == 0: continue
        compared += 1; rtop = int(c[np.argmax(rv[i, c])]); gtop = int(c[np.argmax(gv[i, c])]); changed += int(rtop != gtop); unsafe += int(rtop in set(p.tolist()) and gtop not in set(p.tolist()))
    return {"compared": compared, "unsafe_flip": unsafe, "unsafe_flip_rate": float(unsafe / max(compared, 1)), "top1_change": changed, "top1_change_rate": float(changed / max(compared, 1))}


def prefix_consistency(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    final = vectors["16"]
    # Compare each query with its own prefix-16 vector.  ``np.mean`` must be
    # applied after taking the diagonal; calling ``.diagonal()`` on the scalar
    # mean would either fail or silently mask a contract error.
    return {str(p): float(np.mean(np.diag(vectors[str(p)] @ final.T))) if len(final) else 0.0 for p in PREFIXES[:-1]}


def evaluate_fold(fold: int, metadata: dict[str, dict[str, Any]], features: np.ndarray, model: GroundedCorrespondence) -> dict[str, Any]:
    manifest_path = EPISODE_ROOT / f"episode_manifest_f{fold}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    val_records = [r for r in manifest["records"] if r.get("split") == "val"]
    fit_records = [r for r in manifest["records"] if r.get("split") == "fit"]
    keys = sorted({str(r["query_track_key"]) for r in val_records if str(r.get("query_track_key")) in metadata})
    raw_by_prefix: dict[str, np.ndarray] = {}; grounded_by_prefix: dict[str, np.ndarray] = {}; metrics: dict[str, Any] = {}
    for prefix in PREFIXES:
        raw = np.asarray([raw_vector(k, metadata, features, prefix) for k in keys], dtype=np.float32)
        grounded = np.asarray([grounded_vector(k, metadata, features, prefix, model) for k in keys], dtype=np.float32)
        raw_by_prefix[str(prefix)] = raw; grounded_by_prefix[str(prefix)] = grounded
        metrics[str(prefix)] = {"raw": query_metric(keys, raw, metadata), "grounded": query_metric(keys, grounded, metadata), "paired": paired_unsafe(keys, raw, grounded, metadata)}
    consistency = prefix_consistency(grounded_by_prefix)
    for p in PREFIXES[:-1]: metrics[str(p)]["grounded"]["to_prefix16_cosine"] = consistency[str(p)]
    # Exercise the actual one-to-one correspondence operation on a fixed,
    # deterministic prefix pair; this is a diagnostic and never selects data.
    match_diag = None
    if len(keys) >= 2:
        left = sequence(keys[0], metadata, features, 16); right = sequence(keys[1], metadata, features, 16)
        match_diag = {"query": keys[0], "candidate": keys[1], "score": hungarian_match_score(left, right), "prefix": 16}
    return {
        "fold": fold, "validation_tracklets": len(keys), "fit_records": len(fit_records),
        "fit_positive_episodes": sum(r.get("kind") == "multi_positive_cross_video" for r in fit_records),
        "fit_hard_negative_episodes": sum(r.get("kind") == "null_no_match_hard_negative" for r in fit_records),
        "validation_manifest": str(manifest_path.resolve()), "validation_manifest_sha256": sha256(manifest_path),
        "validation_categories": sorted({metadata[k]["category"] for k in keys}), "validation_videos": sorted({metadata[k]["video"] for k in keys}),
        "prefix": metrics, "grounded_prefix_consistency_to_16": consistency, "hungarian_match_diagnostic": match_diag,
    }


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for prefix in PREFIXES:
        vals = [f["prefix"][str(prefix)] for f in folds]
        out[str(prefix)] = {
            "raw": {m: float(np.mean([v["raw"][m] for v in vals])) for m in ("r1", "r5", "map", "hard_negative_gap", "category_macro_r1", "video_macro_r1")},
            "grounded": {m: float(np.mean([v["grounded"][m] for v in vals])) for m in ("r1", "r5", "map", "hard_negative_gap", "category_macro_r1", "video_macro_r1")},
            "unsafe_flip_rate": float(np.mean([v["paired"]["unsafe_flip_rate"] for v in vals])),
            "unsafe_flip_count": int(sum(v["paired"]["unsafe_flip"] for v in vals)),
            "queries": int(sum(v["grounded"]["queries"] for v in vals)),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=f"phase75c-r-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}")
    args = parser.parse_args()
    rows, tracks, metadata, features, alignment = load_data()
    model = GroundedCorrespondence()
    folds = [evaluate_fold(fold, metadata, features, model) for fold in range(4)]
    agg = aggregate(folds)
    gate_rows = []
    for f in folds:
        raw = f["prefix"]["16"]["raw"]; grounded = f["prefix"]["16"]["grounded"]; pair = f["prefix"]["16"]["paired"]
        gate_rows.append({"fold": f["fold"], "raw_r1": raw["r1"], "grounded_r1": grounded["r1"], "delta_r1": grounded["r1"] - raw["r1"], "raw_map": raw["map"], "grounded_map": grounded["map"], "delta_map": grounded["map"] - raw["map"], "raw_hard_gap": raw["hard_negative_gap"], "grounded_hard_gap": grounded["hard_negative_gap"], "unsafe_flip": pair["unsafe_flip"], "directional": bool(grounded["r1"] > raw["r1"] and grounded["map"] > raw["map"]), "substantial": bool(grounded["r1"] - raw["r1"] >= R1_DELTA and grounded["map"] - raw["map"] >= MAP_DELTA)})
    substantial = sum(int(x["substantial"]) for x in gate_rows); directional = sum(int(x["directional"]) for x in gate_rows); unsafe = sum(int(x["unsafe_flip"]) for x in gate_rows)
    gate_pass = substantial >= 3 and unsafe == 0
    result = {
        "phase": "Phase75C-R", "run_id": args.run_id, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "trackocd_phase75c_grounded_correspondence_train_disjoint_validation", "method": model.metadata(), "training": {"status": "NOT_APPLICABLE_ZERO_PARAMETER_OFFICIAL_METHOD", "reason": "Grounded Correspondence's registered temporal identity operation has zero learnable temporal parameters; no checkpoint or learned weights are introduced."},
        "input_rows": len(rows), "input_tracklets_with_labels": len(metadata), "input_features": str(FEAT_PATH.resolve()), "input_feature_sha256": sha256(FEAT_PATH), "input_csv_sha256": sha256(CSV_PATH), "feature_alignment": alignment,
        "folds": folds, "aggregate": agg, "gate_r": {"thresholds": {"r1_delta": R1_DELTA, "map_delta": MAP_DELTA, "minimum_folds": 3, "unsafe_flip": 0}, "folds_substantial": substantial, "folds_directional": directional, "unsafe_flip_count": unsafe, "fold_rows": gate_rows, "pass": gate_pass, "decision": "P75C_GATE_R_PASS_AUTHORIZE_CONTROLLER" if gate_pass else "P75C_GATE_R_FAIL_STOP_BEFORE_CONTROLLER"},
        "event_evaluator": {"status": "NOT_RUN", "reason": "R uses TRAIN-disjoint validation only; the 152-event evaluator remains reserved for a post-R controller compatibility route."},
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "held 76-event outcomes", "future rows/tracks", "physical/semantic IDs as features", "category/text inputs"],
    }
    atomic_json(OUT / "metrics/r_retrieval.json", result)
    atomic_json(OUT / "audit/gate_rows.json", {"run_id": args.run_id, "gate_rows": gate_rows, "gate": result["gate_r"]})
    atomic_json(OUT / "status.json", {"status": "P75C_GATE_R_PASS" if gate_pass else "P75C_GATE_R_FAIL", "run_id": args.run_id, "gate": result["gate_r"], "outputs": [str(OUT / "metrics/r_retrieval.json"), str(OUT / "audit/gate_rows.json")], "next_action": "authorize unchanged-controller C replay" if gate_pass else "stop this representation route before controller; preserve evidence"})
    atomic_json(OUT / "completion/r_retrieval.done", {"status": result["gate_r"]["decision"], "run_id": args.run_id, "metrics": str(OUT / "metrics/r_retrieval.json")})
    print(json.dumps({"status": result["gate_r"]["decision"], "aggregate_prefix16": agg["16"], "fold_rows": gate_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
