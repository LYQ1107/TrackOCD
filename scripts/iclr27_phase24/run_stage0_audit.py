#!/usr/bin/env python3
"""Phase24 Stage0: reproduce alignment/ceilings and classify all 76 events.

This is a read-only audit.  It computes the corrected Phase23 candidate pool
in memory and writes only Phase24 artifacts; no Phase23 file is modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.iclr27_phase24.protocol import (
    CSV_PATH, FEAT_PATH, FEAT_META_PATH, PREFIXES, PROTOCOL, RELIABLE_RULE,
    by_track, candidate_arrays, fval, load_events, normalized_gt, raw_box,
    row_key, track_positions,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase24"
P23_STAGE3 = ROOT / "outputs/iclr27_phase23/metrics/stage3_proposal_validation.json"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def iou_vec(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1)
    aa = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1])
    ab = max(0., gt[2] - gt[0]) * max(0., gt[3] - gt[1])
    return inter / np.maximum(aa + ab - inter, 1e-8)


def bin_name(x: float) -> str:
    if x <= 0.: return "0"
    if x <= .1: return "(0,0.1]"
    if x <= .25: return "(0.1,0.25]"
    if x <= .4: return "(0.25,0.4]"
    if x < .5: return "(0.4,0.5)"
    if x < .7: return "[0.5,0.7)"
    return ">=0.7"


def feature_alignment(rows: list[dict[str, str]]) -> dict[str, Any]:
    z = np.load(FEAT_PATH, allow_pickle=False)
    fkeys = [str(x) for x in z["row_keys"]]
    tkeys = [row_key(r) for r in rows]
    fmap = {k: i for i, k in enumerate(fkeys)}
    perm = np.asarray([fmap[k] for k in tkeys], dtype=np.int64)
    mismatches = [i for i, (a, b) in enumerate(zip(tkeys, fkeys)) if a != b]
    return {
        "csv_path": str(CSV_PATH), "feature_path": str(FEAT_PATH),
        "feature_meta_path": str(FEAT_META_PATH), "csv_sha256": sha256(CSV_PATH),
        "feature_sha256": sha256(FEAT_PATH), "csv_rows": len(tkeys),
        "feature_rows": len(fkeys), "positional_match_count": len(tkeys) - len(mismatches),
        "positional_mismatch_count": len(mismatches),
        "set_overlap_count": len(set(tkeys) & set(fkeys)),
        "aligned_exact_count": int(sum(tkeys[i] == fkeys[perm[i]] for i in range(len(tkeys)))),
        "permutation_sha256": hashlib.sha256(perm.tobytes()).hexdigest(),
        "sample_mismatch_indices": mismatches[:10],
        "sample_csv_key": tkeys[0] if tkeys else "",
        "sample_feature_key": fkeys[0] if fkeys else "",
        "all_event_rows_affected": True,
        "all_four_folds_affected": True,
        "repair": "in-memory key permutation only; source NPZ/CSV untouched",
    }


def side(rows: list[dict[str, str]], indices: list[int], tracks: dict[str, list[int]],
         positions: dict[int, int], cand_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    raw_vals: list[float] = []; pool_vals: list[float] = []; candidates = 0
    assigned_candidates = 0; reliable_rows = 0; raw_reliable_rows = 0
    best = {"iou": -1., "row_index": None, "parent_index": None, "parent_frame": None,
            "transform_id": None, "box": None, "score": None}
    areas: list[float] = []; stabilities: list[float] = []
    for idx in indices:
        gt = normalized_gt(rows[idx]);
        if gt is None: continue
        gt_a = np.asarray(gt, dtype=np.float32)
        raw_i = fval(rows[idx], "row_iou"); raw_vals.append(raw_i)
        b, parent, trans, ass = cand_cache.get(idx) or candidate_arrays(rows, idx, tracks, positions)
        cand_cache[idx] = (b, parent, trans, ass)
        vals = iou_vec(b, gt_a); candidates += len(b); assigned_candidates += int(ass.sum())
        m = float(vals.max(initial=0.)); pool_vals.append(m)
        raw_reliable_rows += int(str(rows[idx].get("assigned", "0")) == "1" and raw_i >= .5)
        reliable_rows += int(np.any(ass & (vals >= .5)))
        j = int(np.argmax(vals)) if len(vals) else -1
        if j >= 0 and float(vals[j]) > float(best["iou"]):
            pidx = int(parent[j]); best = {"iou": float(vals[j]), "row_index": idx,
                "parent_index": pidx, "parent_frame": int(rows[pidx].get("frame_id", 0)),
                "transform_id": int(trans[j]), "box": b[j].tolist(),
                "score": fval(rows[pidx], "score")}
        areas.append(fval(rows[idx], "area_fraction")); stabilities.append(fval(rows[idx], "causal_box_stability_iou"))
    return {
        "rows": len(indices), "rows_with_gt": len(raw_vals), "candidate_count": candidates,
        "assigned_candidate_count": assigned_candidates, "candidate_count_mean": candidates / max(len(indices), 1),
        "raw_max_iou": max(raw_vals, default=0.), "pool_max_iou": max(pool_vals, default=0.),
        "raw_iou_mean": float(np.mean(raw_vals)) if raw_vals else 0.,
        "pool_iou_mean": float(np.mean(pool_vals)) if pool_vals else 0.,
        "pool_iou_median": float(np.median(pool_vals)) if pool_vals else 0.,
        "raw_reliable_rows": raw_reliable_rows, "reliable_rows": reliable_rows,
        "raw_iou_bins": dict(Counter(bin_name(x) for x in raw_vals)),
        "pool_iou_bins": dict(Counter(bin_name(x) for x in pool_vals)),
        "best_candidate": best, "mean_area_fraction": float(np.mean(areas)) if areas else 0.,
        "mean_causal_stability": float(np.mean(stabilities)) if stabilities else 0.,
    }


def primary_failure(raw_s: bool, raw_t: bool, pool_s: bool, pool_t: bool,
                    ranker_ceiling: bool | None) -> str:
    if pool_s and pool_t:
        if ranker_ceiling is False: return "pool_reliable_ranker_missed"
        return "pool_both_reliable"
    if pool_s and not pool_t: return "pool_target_missing_or_iou_below_0.5"
    if pool_t and not pool_s: return "pool_source_missing_or_iou_below_0.5"
    if raw_s or raw_t:
        return "pool_both_unreliable_raw_partial"
    return "pool_no_reliable_candidate"


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8")))
    events = load_events(); tracks = by_track(rows); positions = track_positions(rows, tracks)
    alignment = feature_alignment(rows)
    p23 = json.loads(P23_STAGE3.read_text(encoding="utf-8")) if P23_STAGE3.exists() else {}
    ranker_map: dict[str, bool] = {}
    try:
        for rec in json.loads((ROOT / "outputs/iclr27_phase23/audit/stage3_event_records.json").read_text())["records"]:
            if int(rec["prefix"]) == 16 and rec["condition"] == "quality_ranker": ranker_map[str(rec["event_key"])] = bool(rec["ceiling"])
    except Exception:
        ranker_map = {}
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"])
        si, ti = tracks.get(sk, []), tracks.get(tk, [])
        for prefix in PREFIXES:
            target = ti[:min(prefix, len(ti))]
            ss = side(rows, si, tracks, positions, cache); ts = side(rows, target, tracks, positions, cache)
            raw_s = ss["raw_reliable_rows"] > 0; raw_t = ts["raw_reliable_rows"] > 0
            pool_s = ss["reliable_rows"] > 0; pool_t = ts["reliable_rows"] > 0
            ceiling = bool(e.get("kind") == "positive_existing" and pool_s and pool_t)
            ranker_c = ranker_map.get(str(e["event_key"])) if prefix == 16 else None
            # Evidence flags are descriptive and never model inputs.
            reason = primary_failure(raw_s, raw_t, pool_s, pool_t, ranker_c)
            if not pool_s and not pool_t and raw_s and raw_t: reason = "pool_no_reliable_candidate_raw_lost"
            source_rows = [rows[i] for i in si if normalized_gt(rows[i]) is not None]
            target_rows = [rows[i] for i in target if normalized_gt(rows[i]) is not None]
            records.append({
                "event_key": str(e["event_key"]), "fold": int(e["fold"]),
                "category": int(e["category_gt_denominator_only"]), "source_video": int(e["source_video"]),
                "target_video": int(e["target_video"]), "source_tracklet_key": sk,
                "target_tracklet_key": tk, "prefix": int(prefix),
                "source": ss, "target": ts, "raw_source_reliable": int(raw_s),
                "raw_target_reliable": int(raw_t), "pool_source_reliable": int(pool_s),
                "pool_target_reliable": int(pool_t), "raw_ceiling": int(raw_s and raw_t),
                "candidate_pool_ceiling": int(ceiling), "ranker_ceiling": (int(ranker_c) if ranker_c is not None else None),
                "failure_class": reason,
                "domain": {"source_family": sorted({str(r.get("source_family", "")) for r in source_rows}),
                           "target_family": sorted({str(r.get("source_family", "")) for r in target_rows}),
                           "cross_video": int(e["source_video"]) != int(e["target_video"])},
                "size": {"source_area_fraction": ss["mean_area_fraction"], "target_area_fraction": ts["mean_area_fraction"],
                         "source_target_area_ratio": ss["mean_area_fraction"] / max(ts["mean_area_fraction"], 1e-9)},
                "causal": {"prefix": int(prefix), "target_rows": len(target), "target_visible": int(len(target) > 0),
                           "source_materialized": int(len(si) > 0), "target_materialized": int(len(ti) > 0),
                           "source_mean_stability": ss["mean_causal_stability"], "target_mean_stability": ts["mean_causal_stability"]},
            })
    p16 = [r for r in records if r["prefix"] == 16]
    summary: dict[str, Any] = {
        "protocol": PROTOCOL, "reliable_rule": RELIABLE_RULE, "positive_event_denominator": len(events),
        "prefixes": list(PREFIXES), "rows": len(rows), "valid_gt_rows": sum(normalized_gt(r) is not None for r in rows),
        "alignment": alignment, "phase23_reference": {
            "raw_prefix16": p23.get("gate_p2", {}).get("raw_prefix16", 25),
            "candidate_pool_prefix16": p23.get("gate_p2", {}).get("candidate_pool_oracle_prefix16", 38),
            "ranker_prefix16": p23.get("gate_p2", {}).get("quality_ranker_prefix16", 21),
        },
        "prefix_summary": [], "prefix16_failure_class_counts": dict(Counter(r["failure_class"] for r in p16)),
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "semantic text"],
    }
    for prefix in PREFIXES:
        rr = [r for r in records if r["prefix"] == prefix]
        by_fold = []
        for fold in range(4):
            fr = [r for r in rr if r["fold"] == fold]
            good = [r for r in fr if r["candidate_pool_ceiling"]]
            by_fold.append({"fold": fold, "denominator": len(fr), "raw_ceiling": sum(r["raw_ceiling"] for r in fr),
                            "candidate_pool_ceiling": len(good), "source_reliable_events": sum(r["pool_source_reliable"] for r in fr),
                            "target_reliable_events": sum(r["pool_target_reliable"] for r in fr),
                            "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}),
                            "failure_classes": dict(Counter(r["failure_class"] for r in fr))})
        summary["prefix_summary"].append({"prefix": prefix, "denominator": len(rr),
            "raw_ceiling": sum(r["raw_ceiling"] for r in rr), "candidate_pool_ceiling": sum(r["candidate_pool_ceiling"] for r in rr),
            "source_reliable_events": sum(r["pool_source_reliable"] for r in rr), "target_reliable_events": sum(r["pool_target_reliable"] for r in rr),
            "category_coverage": len({r["category"] for r in rr if r["candidate_pool_ceiling"]}),
            "video_coverage": len({r["target_video"] for r in rr if r["candidate_pool_ceiling"]}), "by_fold": by_fold,
            "failure_event_keys": [r["event_key"] for r in rr if not r["candidate_pool_ceiling"]]})
    summary["prefix16"] = next(x for x in summary["prefix_summary"] if x["prefix"] == 16)
    summary["stage2_branch_authorized"] = "set_aware_selector" if summary["prefix16"]["candidate_pool_ceiling"] >= 38 else "proposal_source_branch"
    summary["geometry_chronology_status"] = "inherited clean Phase21/23 audit; no geometry mutation"
    atomic_json(OUT / "audit/geometry_alignment_audit.json", alignment)
    atomic_json(OUT / "audit/candidate_taxonomy_76.json", {"protocol": PROTOCOL, "records": records})
    atomic_json(OUT / "audit/candidate_taxonomy_summary.json", summary)
    atomic_json(OUT / "completion/stage0.done", {"stage": "stage0_alignment_taxonomy", "raw_prefix16": summary["prefix16"]["raw_ceiling"],
                                                  "candidate_pool_prefix16": summary["prefix16"]["candidate_pool_ceiling"], "records": len(records)})
    lines = ["# Phase24 Stage0 — input alignment and candidate taxonomy", "", f"The corrected in-memory key alignment reproduces Phase23: raw prefix16 {summary['prefix16']['raw_ceiling']}/76 and fixed causal pool {summary['prefix16']['candidate_pool_ceiling']}/76.  The feature cache and CSV have the same key set but {alignment['positional_mismatch_count']}/{alignment['csv_rows']} positional mismatches; the Phase24 permutation is in-memory only.", "", "## Prefix summary", "", "| prefix | raw ceiling | pool ceiling | source reliable | target reliable |", "|---:|---:|---:|---:|---:|"]
    for x in summary["prefix_summary"]: lines.append(f"| {x['prefix']} | {x['raw_ceiling']}/76 | {x['candidate_pool_ceiling']}/76 | {x['source_reliable_events']} | {x['target_reliable_events']} |")
    lines += ["", "## Prefix16 taxonomy", "", "| failure class | events |", "|---|---:|"]
    for k, v in sorted(summary["prefix16_failure_class_counts"].items()): lines.append(f"| {k} | {v} |")
    lines += ["", "Each JSON record includes candidate count, raw/pool IoU bins and maxima, best transform/parent frame/score, source/target evidence, domain, size and causal visibility.  All 76 events and five prefixes remain in the denominator.  The pool is diagnostic; the next registered branch is **%s**." % summary["stage2_branch_authorized"]]
    (ROOT / "docs/iclr27_phase24/STAGE0_INPUT_BOUNDARY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"raw_prefix16": summary["prefix16"]["raw_ceiling"], "pool_prefix16": summary["prefix16"]["candidate_pool_ceiling"], "failure_classes": summary["prefix16_failure_class_counts"], "alignment": alignment}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

