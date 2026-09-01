#!/usr/bin/env python3
"""Evaluate raw, fixed-pool oracle, trained ranker and GT-tight diagnostics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase23.protocol import CSV_PATH, FEAT_PATH, POS_PATH, PREFIXES, by_track, fval, load_aligned_features, load_events, normalized_gt, raw_box, track_positions
from src.iclr27_phase23.ranker import CandidateQualityRanker
from scripts.iclr27_phase23.train_quality_ranker import TRANSFORM_META, candidate_arrays, feature_batch

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase23"
THR = .5; TOP_K = 5


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0., x2-x1)*max(0., y2-y1); aa = max(0., a[2]-a[0])*max(0., a[3]-a[1]); ab = max(0., b[2]-b[0])*max(0., b[3]-b[1]); return float(inter / max(aa+ab-inter, 1e-8))


def ranker_scores(model: CandidateQualityRanker, rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, idx: int, tracks: dict[str, list[int]], positions: dict[int, int], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes, parent, transform, assigned = candidate_arrays(rows, idx, tracks, positions); current = np.full(len(boxes), idx, dtype=np.int32); v, g = feature_batch(rows, cls, roi, current, parent, boxes, TRANSFORM_META[transform]);
    with torch.no_grad():
        score = model(torch.from_numpy(v).to(device), torch.from_numpy(g).to(device)).float().cpu().numpy()
    return boxes, assigned, score


def main() -> None:
    OUT.joinpath("metrics").mkdir(parents=True, exist_ok=True); OUT.joinpath("audit").mkdir(parents=True, exist_ok=True); OUT.joinpath("completion").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows); tracks = by_track(rows); positions = track_positions(rows, tracks); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--ranker-tag", default="ordered"); args = parser.parse_args()
    rankers: dict[int, CandidateQualityRanker] = {}
    for fold in range(4):
        ckpt = OUT / "checkpoints" / f"ranker_{args.ranker_tag}_f{fold}_best.pt"; ck = torch.load(ckpt, map_location="cpu", weights_only=False); m = CandidateQualityRanker(); m.load_state_dict(ck["model"]); m.to(device).eval(); rankers[fold] = m
    score_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}; pool_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    def pool(idx: int) -> tuple[np.ndarray, np.ndarray]:
        if idx not in pool_cache:
            cand = candidate_arrays(rows, idx, tracks, positions)
            # candidate_arrays returns (boxes, parent_idx, transform_id,
            # parent_assigned); reliability must use the final boolean field,
            # never integer parent IDs.
            pool_cache[idx] = (cand[0], cand[3])
        return pool_cache[idx]
    def side_metrics(indices: list[int], condition: str, fold: int) -> dict[str, Any]:
        ious: list[float] = []; rel_rows = 0; candidate_count = 0; selected_count = 0; assigned_count = 0
        for idx in indices:
            gt = normalized_gt(rows[idx]);
            if condition == "gt_tight_oracle":
                if gt is None: continue
                ious.append(1.0); rel_rows += 1; selected_count += 1; continue
            if condition == "raw_baseline":
                b = np.asarray(raw_box(rows[idx]), dtype=np.float32); boxes = b.reshape(1, 4); ass = np.asarray([str(rows[idx].get("assigned", "0")) == "1"]); scores = None
            else:
                boxes, ass = pool(idx); scores = None
                if condition == "quality_ranker":
                    key = (fold, idx)
                    if key not in score_cache: score_cache[key] = ranker_scores(rankers[fold], rows, cls, roi, idx, tracks, positions, device)
                    boxes, ass, scores = score_cache[key]; order = np.argsort(scores)[::-1][:TOP_K]; boxes, ass = boxes[order], ass[order]; selected_count += len(order)
                else:
                    selected_count += len(boxes)
            candidate_count += len(boxes); assigned_count += int(np.sum(ass))
            if gt is None: continue
            vals = np.asarray([iou(b, np.asarray(gt, dtype=np.float32)) for b in boxes], dtype=np.float32); ious.extend(vals.tolist()); rel_rows += int(np.any(ass & (vals >= THR)))
        return {"rows": len(indices), "candidate_rows": candidate_count, "selected_candidates": selected_count, "assigned_candidates": assigned_count, "reliable_rows": rel_rows, "max_iou": max(ious, default=0.), "iou_mean": float(np.mean(ious)) if ious else 0., "iou_median": float(np.median(ious)) if ious else 0., "iou_values_count": len(ious)}
    conditions = ["raw_baseline", "candidate_pool_oracle", "quality_ranker", "gt_tight_oracle"]; records: list[dict[str, Any]] = []
    for cond in conditions:
        for e in events:
            fold = int(e["fold"]); sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); si, ti = tracks.get(sk, []), tracks.get(tk, [])
            for prefix in PREFIXES:
                sm, tm = side_metrics(si, cond, fold), side_metrics(ti[:min(prefix, len(ti))], cond, fold); ceiling = bool(sm["reliable_rows"] > 0 and tm["reliable_rows"] > 0)
                records.append({"condition": cond, "event_key": str(e["event_key"]), "fold": fold, "category": int(e["category_gt_denominator_only"]), "source_tracklet_key": sk, "target_tracklet_key": tk, "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "source": sm, "target": tm, "source_reliable": int(sm["reliable_rows"] > 0), "target_reliable": int(tm["reliable_rows"] > 0), "ceiling": int(e.get("kind") == "positive_existing" and ceiling), "failure_reasons": (["source_no_reliable"] if sm["reliable_rows"] == 0 else []) + (["target_no_reliable_in_prefix"] if tm["reliable_rows"] == 0 else [])})
    aggregate: dict[str, Any] = {"protocol": "trackocd_iclr27_phase23_stage3_true_iou_candidate_validation", "positive_event_denominator": 76, "prefixes": list(PREFIXES), "reliable_rule": "assigned == 1 and selected candidate true normalized IoU >= 0.5", "top_k_quality_ranker": TOP_K, "conditions": {}, "feature_alignment": alignment, "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels"]}
    for cond in conditions:
        cr = [r for r in records if r["condition"] == cond]; ps = []
        for p in PREFIXES:
            rr = [r for r in cr if r["prefix"] == p]; good = [r for r in rr if r["ceiling"]]; by_fold = []
            for f in range(4):
                fr = [r for r in rr if r["fold"] == f]; fg = [r for r in fr if r["ceiling"]]; by_fold.append({"fold": f, "denominator": len(fr), "source_reliable_events": sum(r["source_reliable"] for r in fr), "target_reliable_events": sum(r["target_reliable"] for r in fr), "ceiling_correct": len(fg), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
            ps.append({"prefix": p, "denominator": len(rr), "source_reliable_events": sum(r["source_reliable"] for r in rr), "target_reliable_events": sum(r["target_reliable"] for r in rr), "ceiling_correct": len(good), "ceiling_recall": len(good)/max(len(rr), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}), "source_iou_mean": float(np.mean([r["source"]["iou_mean"] for r in rr])) if rr else 0., "source_iou_median": float(np.median([r["source"]["iou_median"] for r in rr])) if rr else 0., "target_iou_mean": float(np.mean([r["target"]["iou_mean"] for r in rr])) if rr else 0., "target_iou_median": float(np.median([r["target"]["iou_median"] for r in rr])) if rr else 0., "by_fold": by_fold, "failure_event_keys": [r["event_key"] for r in rr if not r["ceiling"]]})
        aggregate["conditions"][cond] = {"prefix_summary": ps, "prefix16": next(x for x in ps if x["prefix"] == 16), "event_records": len(cr), "diagnostic_only": cond in {"candidate_pool_oracle", "gt_tight_oracle"}}
    r16 = aggregate["conditions"]["raw_baseline"]["prefix16"]; q16 = aggregate["conditions"]["quality_ranker"]["prefix16"]; pool16 = aggregate["conditions"]["candidate_pool_oracle"]["prefix16"]; gt16 = aggregate["conditions"]["gt_tight_oracle"]["prefix16"]; aggregate["gate_p2"] = {"threshold": 38, "raw_prefix16": r16["ceiling_correct"], "candidate_pool_oracle_prefix16": pool16["ceiling_correct"], "quality_ranker_prefix16": q16["ceiling_correct"], "gt_tight_prefix16": gt16["ceiling_correct"], "real_model": "quality_ranker", "pass": bool(q16["ceiling_correct"] >= 38 and q16["source_reliable_events"] > r16["source_reliable_events"] and q16["target_reliable_events"] > r16["target_reliable_events"] and sum(x["ceiling_correct"] > y["ceiling_correct"] for x, y in zip(q16["by_fold"], r16["by_fold"])) >= 3), "decision": "P23_GATE_P2_PASS" if bool(q16["ceiling_correct"] >= 38) else ("P23_GATE_P2_PARTIAL" if q16["ceiling_correct"] >= 30 else "P23_GATE_P2_FAIL")}
    atomic_json(OUT / "metrics/stage3_proposal_validation.json", aggregate); atomic_json(OUT / "audit/stage3_event_records.json", {"protocol": aggregate["protocol"], "records": records})
    fields = ["condition", "event_key", "fold", "category", "prefix", "source_reliable", "target_reliable", "ceiling", "source_iou_mean", "target_iou_mean", "failure_reasons"]; cp = OUT / "audit/stage3_event_summary.csv"; cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({"condition": r["condition"], "event_key": r["event_key"], "fold": r["fold"], "category": r["category"], "prefix": r["prefix"], "source_reliable": r["source_reliable"], "target_reliable": r["target_reliable"], "ceiling": r["ceiling"], "source_iou_mean": r["source"]["iou_mean"], "target_iou_mean": r["target"]["iou_mean"], "failure_reasons": ";".join(r["failure_reasons"])})
    # Complete decisive-prefix event index, one row per event and condition.
    cp16 = OUT / "audit/full_76_event_summary.csv"; cols = ["event_key", "fold", "category"] + [f"{c}_{x}" for c in conditions for x in ("source_reliable", "target_reliable", "ceiling")]; by = {(r["event_key"], r["condition"], r["prefix"]): r for r in records}
    with cp16.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for e in events:
            row = {"event_key": str(e["event_key"]), "fold": int(e["fold"]), "category": int(e["category_gt_denominator_only"])}
            for c in conditions:
                r = by[(str(e["event_key"]), c, 16)]; row.update({f"{c}_source_reliable": r["source_reliable"], f"{c}_target_reliable": r["target_reliable"], f"{c}_ceiling": r["ceiling"]})
            w.writerow(row)
    atomic_json(OUT / "completion/stage3.done", {"stage": "stage3_candidate_validation", "quality_ranker_prefix16": q16["ceiling_correct"], "gate_p2": aggregate["gate_p2"]["decision"]})
    print(json.dumps({"raw": r16["ceiling_correct"], "pool_oracle": pool16["ceiling_correct"], "quality_ranker": q16["ceiling_correct"], "gt_tight": gt16["ceiling_correct"], "gate": aggregate["gate_p2"]["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
