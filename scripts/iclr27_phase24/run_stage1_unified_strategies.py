#!/usr/bin/env python3
"""Phase24 Stage1 unified causal candidate-selection diagnostics.

No parameters are fitted here.  The script evaluates the frozen raw stream,
fixed causal score/history ranking, the corrected Phase23 MLP, top-K retention
and a pre-registered uncertainty fallback on the same 76-event denominator.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase24.protocol import (
    CSV_PATH, FEAT_PATH, PREFIXES, PROTOCOL, RELIABLE_RULE, TRANSFORM_META,
    by_track, candidate_arrays, fval, load_aligned_features, load_events,
    normalized_gt, raw_box, track_positions,
)
from src.iclr27_phase23.ranker import CandidateQualityRanker
from scripts.iclr27_phase23.train_quality_ranker import feature_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase24"
THR = 0.5
TOP_KS = (1, 5, 10, 20)


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


def history_consistency(rows: list[dict[str, str]], parent: np.ndarray,
                        tracks: dict[str, list[int]], positions: dict[int, int]) -> np.ndarray:
    vals = np.zeros(len(parent), dtype=np.float32)
    raw_cache: dict[int, np.ndarray] = {}
    for j, p in enumerate(parent.tolist()):
        p = int(p); r = rows[p]
        key = f"v{int(r['video_id'])}:p{int(r['track_id'])}"; inds = tracks[key]; pos = positions[p]
        past = inds[max(0, pos - 4):pos]
        if not past: vals[j] = 0.; continue
        b = np.asarray(raw_box(r), dtype=np.float32)
        scores = []
        for q in past:
            if q not in raw_cache: raw_cache[q] = np.asarray(raw_box(rows[q]), dtype=np.float32)
            scores.append(float(iou_vec(b[None, :], raw_cache[q])[0]))
        vals[j] = float(np.mean(scores)) if scores else 0.
    return vals


def load_rankers(device: torch.device) -> dict[int, CandidateQualityRanker]:
    out: dict[int, CandidateQualityRanker] = {}
    for fold in range(4):
        p = ROOT / f"outputs/iclr27_phase23/checkpoints/ranker_ordered_f{fold}_best.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = CandidateQualityRanker(); m.load_state_dict(ck["model"]); m.to(device).eval(); out[fold] = m
    return out


def score_candidates(model: CandidateQualityRanker, rows: list[dict[str, str]], cls: np.ndarray,
                     roi: np.ndarray, idx: int, tracks: dict[str, list[int]],
                     positions: dict[int, int], device: torch.device,
                     cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if idx not in cache: cache[idx] = candidate_arrays(rows, idx, tracks, positions)
    boxes, parent, trans, assigned = cache[idx]
    current = np.full(len(boxes), idx, dtype=np.int32)
    v, g = feature_batch(rows, cls, roi, current, parent, boxes, TRANSFORM_META[trans])
    with torch.no_grad(): scores = model(torch.from_numpy(v).to(device), torch.from_numpy(g).to(device)).float().cpu().numpy()
    h = history_consistency(rows, parent, tracks, positions)
    return boxes, parent, trans, assigned, scores, h


def select_condition(condition: str, idx: int, fold: int, rows: list[dict[str, str]], cls: np.ndarray,
                     roi: np.ndarray, tracks: dict[str, list[int]], positions: dict[int, int],
                     rankers: dict[int, CandidateQualityRanker], device: torch.device,
                     cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    r = rows[idx]
    if condition == "raw_baseline":
        return {"boxes": np.asarray(raw_box(r), dtype=np.float32).reshape(1, 4), "parent": np.asarray([idx]),
                "transform": np.asarray([-1]), "assigned": np.asarray([str(r.get("assigned", "0")) == "1"]),
                "scores": np.asarray([fval(r, "score")], dtype=np.float32), "deferred": False}
    if condition == "gt_tight_oracle":
        gt = normalized_gt(r)
        return {"boxes": np.asarray(gt if gt is not None else raw_box(r), dtype=np.float32).reshape(1, 4), "parent": np.asarray([idx]),
                "transform": np.asarray([-2]), "assigned": np.asarray([True]), "scores": np.asarray([1.], dtype=np.float32), "deferred": False}
    if idx not in cache: cache[idx] = candidate_arrays(rows, idx, tracks, positions)
    boxes, parent, trans, assigned = cache[idx]
    if condition == "candidate_pool_oracle":
        return {"boxes": boxes, "parent": parent, "transform": trans, "assigned": assigned,
                "scores": np.asarray([fval(rows[int(p)], "score") for p in parent], dtype=np.float32), "deferred": False}
    model = rankers[fold]
    b, p, t, a, scores, hist = score_candidates(model, rows, cls, roi, idx, tracks, positions, device, cache)
    if condition == "fixed_combo":
        raw_scores = np.asarray([fval(rows[int(q)], "score") for q in p], dtype=np.float32)
        combo = .55 * raw_scores + .25 * np.asarray([fval(rows[int(q)], "causal_box_stability_iou") for q in p], dtype=np.float32) + .20 * hist
        order = np.argsort(combo)[::-1][:1]
        return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": combo[order], "deferred": False}
    order = np.argsort(scores)[::-1]
    if condition.startswith("phase23_mlp_top"):
        k = int(condition.rsplit("top", 1)[1]); order = order[:min(k, len(order))]
        return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": scores[order], "deferred": False}
    if condition == "uncertainty_defer":
        top = order[:1]; conf = float(1. / (1. + np.exp(-scores[top[0]]))) if len(top) else 0.
        margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float("inf")
        defer = conf < .50 or margin < .05
        if defer:
            # Explicitly union raw with the MLP choice; raw is never discarded.
            rb = np.asarray(raw_box(r), dtype=np.float32).reshape(1, 4)
            return {"boxes": np.concatenate([rb, b[top]], axis=0), "parent": np.concatenate([np.asarray([idx]), p[top]]),
                    "transform": np.concatenate([np.asarray([-1]), t[top]]), "assigned": np.concatenate([np.asarray([str(r.get("assigned", "0")) == "1"]), a[top]]),
                    "scores": np.concatenate([np.asarray([fval(r, "score")]), scores[top]]), "deferred": True,
                    "confidence": conf, "margin": margin}
        return {"boxes": b[top], "parent": p[top], "transform": t[top], "assigned": a[top], "scores": scores[top], "deferred": False,
                "confidence": conf, "margin": margin}
    raise ValueError(condition)


def side_eval(indices: list[int], condition: str, fold: int, rows: list[dict[str, str]], cls: np.ndarray,
              roi: np.ndarray, tracks: dict[str, list[int]], positions: dict[int, int], rankers: dict[int, CandidateQualityRanker],
              device: torch.device, cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    all_ious: list[float] = []; max_ious: list[float] = []; rel_rows = 0; recalls = {str(t): 0 for t in (.3, .5, .7)}; total_rows = 0; deferred = 0
    selected = 0; assigned_count = 0; evidence: list[dict[str, Any]] = []
    for idx in indices:
        gt = normalized_gt(rows[idx]);
        if gt is None: continue
        total_rows += 1; sel = select_condition(condition, idx, fold, rows, cls, roi, tracks, positions, rankers, device, cache); deferred += int(sel.get("deferred", False)); selected += len(sel["boxes"]); assigned_count += int(np.sum(sel["assigned"]));
        vals = iou_vec(np.asarray(sel["boxes"], dtype=np.float32), np.asarray(gt, dtype=np.float32)); all_ious.extend(vals.tolist()); max_i = float(vals.max(initial=0.)); max_ious.append(max_i)
        for t in recalls:
            recalls[t] += int(np.any(sel["assigned"] & (vals >= float(t))))
        if len(vals):
            j = int(np.argmax(vals)); evidence.append({"row_index": int(idx), "selected_count": len(vals), "max_iou": max_i,
                "best_iou": float(vals[j]), "best_parent_index": int(sel["parent"][j]), "best_transform": int(sel["transform"][j]),
                "best_parent_frame": int(rows[int(sel["parent"][j])].get("frame_id", 0)), "deferred": bool(sel.get("deferred", False)),
                "confidence": sel.get("confidence"), "margin": sel.get("margin")})
    return {"rows": len(indices), "rows_with_gt": total_rows, "selected_candidates": selected, "assigned_candidates": assigned_count,
            "reliable_rows": rel_rows if False else recalls["0.5"], "recall_at_0.3_rows": recalls["0.3"], "recall_at_0.5_rows": recalls["0.5"], "recall_at_0.7_rows": recalls["0.7"],
            "recall_at_0.3": recalls["0.3"] / max(total_rows, 1), "recall_at_0.5": recalls["0.5"] / max(total_rows, 1), "recall_at_0.7": recalls["0.7"] / max(total_rows, 1),
            "max_iou_mean": float(np.mean(max_ious)) if max_ious else 0., "max_iou_median": float(np.median(max_ious)) if max_ious else 0.,
            "iou_mean": float(np.mean(all_ious)) if all_ious else 0., "iou_median": float(np.median(all_ious)) if all_ious else 0.,
            "deferred_rows": deferred, "evidence": evidence}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows)
    tracks = by_track(rows); positions = track_positions(rows, tracks); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rankers = load_rankers(device); cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    conditions = ["raw_baseline", "fixed_combo", "phase23_mlp_top1", "phase23_mlp_top5", "phase23_mlp_top10", "phase23_mlp_top20", "uncertainty_defer", "candidate_pool_oracle", "gt_tight_oracle"]
    records: list[dict[str, Any]] = []
    for cond in conditions:
        for e in events:
            fold = int(e["fold"]); sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); si, ti = tracks.get(sk, []), tracks.get(tk, [])
            for prefix in PREFIXES:
                sm = side_eval(si, cond, fold, rows, cls, roi, tracks, positions, rankers, device, cache)
                tm = side_eval(ti[:min(prefix, len(ti))], cond, fold, rows, cls, roi, tracks, positions, rankers, device, cache)
                src = sm["reliable_rows"] > 0; tgt = tm["reliable_rows"] > 0
                records.append({"condition": cond, "event_key": str(e["event_key"]), "fold": fold, "category": int(e["category_gt_denominator_only"]),
                    "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "source": sm, "target": tm,
                    "source_reliable": int(src), "target_reliable": int(tgt), "ceiling": int(e.get("kind") == "positive_existing" and src and tgt),
                    "deferred_source_rows": sm["deferred_rows"], "deferred_target_rows": tm["deferred_rows"],
                    "failure_reasons": ([] if src else ["source_no_reliable_selected"]) + ([] if tgt else ["target_no_reliable_selected_in_prefix"]),
                    "diagnostic_only": cond in {"candidate_pool_oracle", "gt_tight_oracle"}})
    aggregate: dict[str, Any] = {"protocol": PROTOCOL + "_stage1_unified", "positive_event_denominator": len(events), "prefixes": list(PREFIXES),
        "reliable_rule": RELIABLE_RULE, "conditions": {}, "feature_alignment": alignment, "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH),
        "feature_sha256": sha256(FEAT_PATH), "registered_constants": {"fixed_combo": {"raw_score": .55, "stability": .25, "history_consistency": .20}, "uncertainty_confidence": .50, "uncertainty_margin": .05, "top_ks": list(TOP_KS)},
        "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "semantic text"]}
    for cond in conditions:
        cr = [r for r in records if r["condition"] == cond]; ps = []
        for p in PREFIXES:
            rr = [r for r in cr if r["prefix"] == p]; good = [r for r in rr if r["ceiling"]]; folds = []
            for fold in range(4):
                fr = [r for r in rr if r["fold"] == fold]; fg = [r for r in fr if r["ceiling"]]
                folds.append({"fold": fold, "denominator": len(fr), "source_reliable_events": sum(r["source_reliable"] for r in fr), "target_reliable_events": sum(r["target_reliable"] for r in fr),
                              "ceiling_correct": len(fg), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
            ps.append({"prefix": p, "denominator": len(rr), "source_reliable_events": sum(r["source_reliable"] for r in rr), "target_reliable_events": sum(r["target_reliable"] for r in rr),
                       "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rr), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}),
                       "source_iou_mean": float(np.mean([r["source"]["max_iou_mean"] for r in rr])) if rr else 0., "target_iou_mean": float(np.mean([r["target"]["max_iou_mean"] for r in rr])) if rr else 0.,
                       "source_recall_at_0.3": float(np.mean([r["source"]["recall_at_0.3"] for r in rr])) if rr else 0., "target_recall_at_0.3": float(np.mean([r["target"]["recall_at_0.3"] for r in rr])) if rr else 0.,
                       "source_recall_at_0.5": float(np.mean([r["source"]["recall_at_0.5"] for r in rr])) if rr else 0., "target_recall_at_0.5": float(np.mean([r["target"]["recall_at_0.5"] for r in rr])) if rr else 0.,
                       "source_recall_at_0.7": float(np.mean([r["source"]["recall_at_0.7"] for r in rr])) if rr else 0., "target_recall_at_0.7": float(np.mean([r["target"]["recall_at_0.7"] for r in rr])) if rr else 0.,
                       "deferred_rows": sum(r["deferred_source_rows"] + r["deferred_target_rows"] for r in rr), "by_fold": folds,
                       "failure_event_keys": [r["event_key"] for r in rr if not r["ceiling"]]})
        aggregate["conditions"][cond] = {"prefix_summary": ps, "prefix16": next(x for x in ps if x["prefix"] == 16), "event_records": len(cr), "diagnostic_only": cond in {"candidate_pool_oracle", "gt_tight_oracle"}}
    raw = aggregate["conditions"]["raw_baseline"]["prefix16"]; pool = aggregate["conditions"]["candidate_pool_oracle"]["prefix16"]
    best_real = max((c for c in conditions if c not in {"candidate_pool_oracle", "gt_tight_oracle"}), key=lambda c: aggregate["conditions"][c]["prefix16"]["ceiling_correct"])
    aggregate["stage2_authorization"] = {"candidate_pool_prefix16": pool["ceiling_correct"], "threshold": 38, "authorized": bool(pool["ceiling_correct"] >= 38), "branch": "set_aware_selector" if pool["ceiling_correct"] >= 38 else "proposal_source_branch", "best_stage1_real": best_real, "best_stage1_real_prefix16": aggregate["conditions"][best_real]["prefix16"]["ceiling_correct"]}
    atomic_json(OUT / "metrics/stage1_unified_strategies.json", aggregate); atomic_json(OUT / "audit/stage1_strategy_event_records.json", {"protocol": aggregate["protocol"], "records": records})
    fields = ["condition", "event_key", "fold", "category", "prefix", "source_reliable", "target_reliable", "ceiling", "source_recall_at_0.5", "target_recall_at_0.5", "failure_reasons"]
    with (OUT / "audit/stage1_strategy_event_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({"condition": r["condition"], "event_key": r["event_key"], "fold": r["fold"], "category": r["category"], "prefix": r["prefix"], "source_reliable": r["source_reliable"], "target_reliable": r["target_reliable"], "ceiling": r["ceiling"], "source_recall_at_0.5": r["source"]["recall_at_0.5"], "target_recall_at_0.5": r["target"]["recall_at_0.5"], "failure_reasons": ";".join(r["failure_reasons"])})
    atomic_json(OUT / "completion/stage1.done", {"stage": "stage1_unified_strategies", "raw_prefix16": raw["ceiling_correct"], "pool_prefix16": pool["ceiling_correct"], "best_real": best_real, "best_real_prefix16": aggregate["conditions"][best_real]["prefix16"]["ceiling_correct"]})
    lines = ["# Phase24 Stage1 — unified causal candidate-selection diagnostics", "", "All conditions use the corrected key-aligned feature order and the unchanged 76-event/causal-prefix protocol.  Oracle rows are diagnostics only.", "", "| condition | p1 | p2 | p4 | p8 | p16 | diagnostic |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for c in conditions:
        ps = aggregate["conditions"][c]["prefix_summary"]; vals = [next(x["ceiling_correct"] for x in ps if x["prefix"] == p) for p in PREFIXES]
        lines.append(f"| {c} | " + " | ".join(f"{v}/76" for v in vals) + f" | {'yes' if aggregate['conditions'][c]['diagnostic_only'] else 'no'} |")
    lines += ["", f"The fixed pool remains {pool['ceiling_correct']}/76 at prefix16 (raw {raw['ceiling_correct']}/76), so the registered Stage2 branch is **{aggregate['stage2_authorization']['branch']}**.  The best no-training real strategy is `{best_real}` at {aggregate['stage2_authorization']['best_stage1_real_prefix16']}/76.  Full event records and recall/top-K evidence are in `outputs/iclr27_phase24/audit/stage1_strategy_event_records.json` and `outputs/iclr27_phase24/metrics/stage1_unified_strategies.json`."]
    (ROOT / "docs/iclr27_phase24/STAGE1_UNIFIED_STRATEGY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"raw_prefix16": raw["ceiling_correct"], "pool_prefix16": pool["ceiling_correct"], "best_real": best_real, "best_real_prefix16": aggregate["stage2_authorization"]["best_stage1_real_prefix16"], "branch": aggregate["stage2_authorization"]["branch"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
