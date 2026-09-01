#!/usr/bin/env python3
"""Phase23 Stage1 fixed causal multi-candidate proposal pool oracle."""
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

from src.iclr27_phase23.protocol import (CSV_PATH, IOU_THRESHOLDS, MAX_HISTORY,
    POS_PATH, PREFIXES, SCALE_FACTORS, CENTER_SHIFTS, by_track,
    fixed_transforms, fval, load_events, normalized_gt, raw_box,
    track_positions)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase23"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def bin_name(x: float) -> str:
    if x == 0: return "0"
    if x <= .1: return "(0,0.1]"
    if x <= .25: return "(0.1,0.25]"
    if x <= .4: return "(0.25,0.4]"
    if x < .5: return "(0.4,0.5)"
    if x < .7: return "[0.5,0.7)"
    return ">=0.7"


def candidate_arrays(rows: list[dict[str, str]], idx: int, tracks: dict[str, list[int]], positions: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create compact boxes/parent-assignment arrays for one causal row."""
    key = f"v{int(rows[idx]['video_id'])}:p{int(rows[idx]['track_id'])}"
    inds = tracks[key]; pos = positions[idx]; hist = inds[max(0, pos - MAX_HISTORY + 1):pos + 1]
    base = np.asarray([raw_box(rows[j]) for j in hist], dtype=np.float32); vals = []
    for scale in SCALE_FACTORS:
        for dx in CENTER_SHIFTS:
            for dy in CENTER_SHIFTS:
                cx = (base[:, 0] + base[:, 2]) * .5 + dx * (base[:, 2] - base[:, 0]); cy = (base[:, 1] + base[:, 3]) * .5 + dy * (base[:, 3] - base[:, 1])
                bw = np.maximum(0., base[:, 2] - base[:, 0]) * scale; bh = np.maximum(0., base[:, 3] - base[:, 1]) * scale
                vals.append(np.stack([cx - bw*.5, cy - bh*.5, cx + bw*.5, cy + bh*.5], axis=1))
    boxes = np.clip(np.concatenate(vals, axis=0), 0., 1.)
    # Boxes are concatenated transform-major (all history parents for each
    # transform), hence parent metadata must use the same ordering.
    assigned = np.tile(np.asarray([str(rows[j].get("assigned", "0")) == "1" for j in hist], dtype=bool), len(SCALE_FACTORS) * len(CENTER_SHIFTS) * len(CENTER_SHIFTS))
    return boxes, assigned


def iou_vector(boxes: np.ndarray, gt: list[float] | np.ndarray) -> np.ndarray:
    b = np.asarray(gt, dtype=np.float32); x1 = np.maximum(boxes[:, 0], b[0]); y1 = np.maximum(boxes[:, 1], b[1]); x2 = np.minimum(boxes[:, 2], b[2]); y2 = np.minimum(boxes[:, 3], b[3]); inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1); aa = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1]); ab = max(0., b[2] - b[0]) * max(0., b[3] - b[1]); return inter / np.maximum(aa + ab - inter, 1e-8)


def side(rows: list[dict[str, str]], indices: list[int], cache: dict[int, tuple[np.ndarray, np.ndarray]], tracks: dict[str, list[int]], positions: dict[int, int]) -> dict[str, Any]:
    vals: list[float] = []; raw_vals: list[float] = []; reliable = raw_rel = pool_n = gt_n = 0
    for idx in indices:
        gt = normalized_gt(rows[idx]); pair = cache.get(idx)
        boxes, assigned = pair if pair is not None else candidate_arrays(rows, idx, tracks, positions); pool_n += len(boxes)
        if gt is None: continue
        gt_n += 1; raw_i = fval(rows[idx], "row_iou"); raw_vals.append(raw_i); ious = iou_vector(boxes, gt); vals.append(float(np.max(ious, initial=0.0))); reliable += int(np.any(assigned & (ious >= .5))); raw_rel += int(str(rows[idx].get("assigned", "0")) == "1" and raw_i >= .5)
    return {"rows": len(indices), "rows_with_gt": gt_n, "candidate_box_count": pool_n, "candidate_box_count_mean": pool_n / max(len(indices), 1), "raw_max_iou": max(raw_vals, default=0.), "pool_max_iou": max(vals, default=0.), "raw_iou_mean": float(np.mean(raw_vals)) if raw_vals else 0., "pool_iou_mean": float(np.mean(vals)) if vals else 0., "pool_iou_median": float(np.median(vals)) if vals else 0., "raw_reliable_rows": raw_rel, "reliable_rows": reliable, "pool_iou_bins": dict(Counter(bin_name(x) for x in vals))}


def main() -> None:
    OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); tracks = by_track(rows); positions = track_positions(rows, tracks)
    event_row_indices: set[int] = set()
    for e in events:
        event_row_indices.update(tracks.get(str(e["source_tracklet_keys"][0]), [])); event_row_indices.update(tracks.get(str(e["target_tracklet_key"]), []))
    cache = {i: candidate_arrays(rows, i, tracks, positions) for i in event_row_indices}
    train_gt = [i for i, r in enumerate(rows) if normalized_gt(r) is not None]
    train_recall: dict[str, Any] = {}
    for thr in IOU_THRESHOLDS:
        raw_n = pool_n = 0; by_video: dict[int, list[int]] = defaultdict(list); by_cat: dict[int, list[int]] = defaultdict(list)
        for i in train_gt:
            gt = normalized_gt(rows[i]); boxes, assigned = candidate_arrays(rows, i, tracks, positions); hits = iou_vector(boxes, gt); raw_hit = fval(rows[i], "row_iou") >= thr and str(rows[i].get("assigned", "0")) == "1"; pool_hit = bool(np.any(assigned & (hits >= thr))); raw_n += int(raw_hit); pool_n += int(pool_hit); v, c = int(rows[i]["video_id"]), int(rows[i].get("gt_category_id_common", -1)); by_video[v].append(int(pool_hit)); by_cat[c].append(int(pool_hit))
        train_recall[str(thr)] = {"gt_rows": len(train_gt), "raw_recall": raw_n / max(len(train_gt), 1), "candidate_pool_recall": pool_n / max(len(train_gt), 1), "raw_hits": raw_n, "candidate_pool_hits": pool_n, "video_macro_recall": float(np.mean([np.mean(x) for x in by_video.values()])) if by_video else 0., "category_macro_recall": float(np.mean([np.mean(x) for x in by_cat.values()])) if by_cat else 0.}
    records: list[dict[str, Any]] = []
    for e in events:
        sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); si, ti = tracks.get(sk, []), tracks.get(tk, [])
        for prefix in PREFIXES:
            tgt = ti[:min(prefix, len(ti))]; ss = side(rows, si, cache, tracks, positions); ts = side(rows, tgt, cache, tracks, positions); src_ok, tgt_ok = ss["reliable_rows"] > 0, ts["reliable_rows"] > 0; raw_src, raw_tgt = ss["raw_reliable_rows"] > 0, ts["raw_reliable_rows"] > 0; failures = []
            if not src_ok: failures.append("source_pool_no_reliable_observation")
            if not tgt_ok: failures.append("target_pool_no_reliable_observation_in_prefix")
            records.append({"event_key": str(e["event_key"]), "fold": int(e["fold"]), "category": int(e["category_gt_denominator_only"]), "source_tracklet_key": sk, "target_tracklet_key": tk, "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "source": ss, "target": ts, "raw_source_reliable": int(raw_src), "raw_target_reliable": int(raw_tgt), "raw_ceiling": bool(raw_src and raw_tgt), "candidate_pool_ceiling": bool(e.get("kind") == "positive_existing" and src_ok and tgt_ok), "failure_reasons": failures})
    summary: dict[str, Any] = {"protocol": "trackocd_iclr27_phase23_stage1_fixed_causal_candidate_pool_oracle", "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "positive_event_denominator": len(events), "prefixes": list(PREFIXES), "reliable_rule": "candidate parent assigned == 1 and true normalized IoU >= 0.5", "transform_grid": {"scale_factors": list(SCALE_FACTORS), "center_shifts": list(CENTER_SHIFTS), "max_history": MAX_HISTORY, "transforms_per_parent": len(fixed_transforms([.1, .1, .2, .2]))}, "all_raw_rows_retained": True, "train_gt_candidate_recall": train_recall, "prefix_summary": [], "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    for p in PREFIXES:
        rr = [r for r in records if r["prefix"] == p]; good = [r for r in rr if r["candidate_pool_ceiling"]]; fs = []
        for f in range(4):
            fr = [r for r in rr if r["fold"] == f]; fg = [r for r in fr if r["candidate_pool_ceiling"]]; fs.append({"fold": f, "denominator": len(fr), "source_reliable_events": sum(r["source"]["reliable_rows"] > 0 for r in fr), "target_reliable_events": sum(r["target"]["reliable_rows"] > 0 for r in fr), "raw_ceiling": sum(r["raw_ceiling"] for r in fr), "candidate_pool_ceiling": len(fg), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
        summary["prefix_summary"].append({"prefix": p, "denominator": len(rr), "raw_ceiling": sum(r["raw_ceiling"] for r in rr), "candidate_pool_ceiling": len(good), "candidate_pool_recall": len(good) / max(len(rr), 1), "source_reliable_events": sum(r["source"]["reliable_rows"] > 0 for r in rr), "target_reliable_events": sum(r["target"]["reliable_rows"] > 0 for r in rr), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}), "source_iou_mean": float(np.mean([r["source"]["pool_iou_mean"] for r in rr])), "target_iou_mean": float(np.mean([r["target"]["pool_iou_mean"] for r in rr])), "by_fold": fs, "failure_event_keys": [r["event_key"] for r in rr if not r["candidate_pool_ceiling"]]})
    summary["prefix16"] = next(x for x in summary["prefix_summary"] if x["prefix"] == 16); summary["stage2_branch"] = "stage2a_quality_ranker" if summary["prefix16"]["candidate_pool_ceiling"] >= 38 else "stage2b_proposal_source_branch"; summary["gate_o_candidate_pool"] = bool(summary["prefix16"]["candidate_pool_ceiling"] >= 38 and sum(x["candidate_pool_ceiling"] > x["raw_ceiling"] for x in summary["prefix16"]["by_fold"]) >= 3)
    atomic_json(OUT / "audit/candidate_pool_recall.json", summary); atomic_json(OUT / "audit/candidate_event_records.json", {"protocol": summary["protocol"], "records": records}); atomic_json(OUT / "completion/stage1.done", {"stage": "stage1_candidate_pool", "prefix16_candidate_pool_ceiling": summary["prefix16"]["candidate_pool_ceiling"], "stage2_branch": summary["stage2_branch"], "gate_o_candidate_pool": summary["gate_o_candidate_pool"]})
    lines = ["# Phase23 Stage 1 — fixed causal candidate-pool oracle", "", "Every raw DSCT row is retained. Each current row receives 27 pre-registered scale/center transforms and the same transforms of up to four earlier same-track boxes; no future row, event-specific choice, or GT box enters the candidate set.", "", "## TRAIN-only recall", "", "| IoU threshold | GT rows | raw hits | pool hits | raw recall | pool recall |", "|---:|---:|---:|---:|---:|---:|"]
    for t, x in train_recall.items(): lines.append(f"| {t} | {x['gt_rows']} | {x['raw_hits']} | {x['candidate_pool_hits']} | {x['raw_recall']:.4f} | {x['candidate_pool_recall']:.4f} |")
    lines += ["", "## 76-event oracle", "", "| prefix | raw ceiling | pool ceiling | source events | target events | category coverage | video coverage |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for x in summary["prefix_summary"]: lines.append(f"| {x['prefix']} | {x['raw_ceiling']}/76 | {x['candidate_pool_ceiling']}/76 | {x['source_reliable_events']} | {x['target_reliable_events']} | {x['category_coverage']} | {x['video_coverage']} |")
    lines += ["", f"At prefix16 the fixed pool reaches **{summary['prefix16']['candidate_pool_ceiling']}/76**; the registered next branch is **{summary['stage2_branch']}**. The pool is an oracle diagnostic only and does not rank candidates or alter the online evaluator. Full event/prefix evidence is in [`candidate_event_records.json`](../../outputs/iclr27_phase23/audit/candidate_event_records.json) and machine metrics in [`candidate_pool_recall.json`](../../outputs/iclr27_phase23/audit/candidate_pool_recall.json)."]
    (ROOT / "docs/iclr27_phase23/CANDIDATE_POOL_ABLATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"prefix16_raw": summary["prefix16"]["raw_ceiling"], "prefix16_pool": summary["prefix16"]["candidate_pool_ceiling"], "stage2_branch": summary["stage2_branch"], "gate_o_candidate_pool": summary["gate_o_candidate_pool"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
