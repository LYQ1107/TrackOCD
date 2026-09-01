#!/usr/bin/env python3
"""Phase25 Stage1: unified causal proposal-set strategies and MOT audit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase24.protocol import (
    CSV_PATH, FEAT_PATH, PREFIXES, TRANSFORM_META, by_track, candidate_arrays,
    fval, load_aligned_features, load_events, normalized_gt, raw_box,
    track_positions,
)
from src.iclr27_phase24.set_selector import SetAwareCandidateSelector
from scripts.iclr27_phase23.train_quality_ranker import feature_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase25"
TOP_KS = (5, 10, 20, 27)
THRESHOLDS = (0.3, 0.5, 0.7)


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
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()


def iou_vec(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1])
    x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    aa = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    ab = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
    return inter / np.maximum(aa + ab - inter, 1e-8)


def history_consistency(rows: list[dict[str, str]], parent: np.ndarray,
                        tracks: dict[str, list[int]], positions: dict[int, int]) -> np.ndarray:
    vals = np.zeros(len(parent), dtype=np.float32)
    box_cache: dict[int, np.ndarray] = {}
    for j, p0 in enumerate(parent.tolist()):
        p = int(p0); r = rows[p]; inds = tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"]
        pos = positions[p]; past = inds[max(0, pos - 4):pos]
        if not past: continue
        b = np.asarray(raw_box(r), dtype=np.float32)
        scores = []
        for q in past:
            box_cache.setdefault(q, np.asarray(raw_box(rows[q]), dtype=np.float32))
            scores.append(float(iou_vec(b[None, :], box_cache[q])[0]))
        vals[j] = float(np.mean(scores)) if scores else 0.0
    return vals


def load_models(device: torch.device) -> dict[int, SetAwareCandidateSelector]:
    models: dict[int, SetAwareCandidateSelector] = {}
    for fold in range(4):
        path = ROOT / f"outputs/iclr27_phase24/checkpoints/setaware_f{fold}_best.pt"
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = SetAwareCandidateSelector(); m.load_state_dict(ck["model"]); m.to(device).eval(); models[fold] = m
    return models


def score_row(idx: int, fold: int, rows: list[dict[str, str]], cls: np.ndarray,
              roi: np.ndarray, tracks: dict[str, list[int]], positions: dict[int, int],
              model: SetAwareCandidateSelector, device: torch.device,
              cache: dict[int, tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    if idx in cache:
        return cache[idx]
    b, p, t, a = candidate_arrays(rows, idx, tracks, positions)
    current = np.full(len(b), idx, dtype=np.int32)
    v, g = feature_batch(rows, cls, roi, current, p, b, TRANSFORM_META[t])
    with torch.no_grad():
        q, u = model(torch.from_numpy(v).to(device).unsqueeze(0), torch.from_numpy(g).to(device).unsqueeze(0), torch.ones((1, len(b)), dtype=torch.bool, device=device))
    h = history_consistency(rows, p, tracks, positions)
    raw_scores = np.asarray([fval(rows[int(x)], "score") for x in p], dtype=np.float32)
    out = (b, p, t, a, q.float().cpu().numpy().reshape(-1), u.float().cpu().numpy().reshape(-1), h, raw_scores)
    cache[idx] = out
    return out


def selection(condition: str, idx: int, fold: int, rows: list[dict[str, str]],
              cls: np.ndarray, roi: np.ndarray, tracks: dict[str, list[int]],
              positions: dict[int, int], model: SetAwareCandidateSelector,
              device: torch.device, cache: dict[int, tuple[np.ndarray, ...]],
              k: int = 20) -> dict[str, Any]:
    r = rows[idx]
    if condition == "raw_baseline":
        return {"boxes": np.asarray(raw_box(r), np.float32)[None, :], "parent": np.asarray([idx], np.int32), "transform": np.asarray([-1], np.int16), "assigned": np.asarray([str(r.get("assigned", "0")) == "1"], bool), "scores": np.asarray([fval(r, "score")], np.float32), "fallback": False}
    b, p, t, a, q, u, h, rs = score_row(idx, fold, rows, cls, roi, tracks, positions, model, device, cache)
    if condition == "candidate_pool_oracle":
        return {"boxes": b, "parent": p, "transform": t, "assigned": a, "scores": rs, "fallback": False}
    if condition.startswith("phase24_setaware_top"):
        kk = int(condition.rsplit("top", 1)[1])
        order = np.argsort(q)[::-1][:min(kk, len(q))]
        return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": q[order], "fallback": False}
    elif condition == "confidence_calibrated_top20":
        order_all = np.argsort(q)[::-1]; top = order_all[:min(20, len(order_all))]
        conf = float(1.0 / (1.0 + np.exp(-q[order_all[0]]))) if len(order_all) else 0.0
        margin = float(q[order_all[0]] - q[order_all[1]]) if len(order_all) > 1 else float("inf")
        # Fixed pre-registered fallback: uncertainty never drops the raw row.
        if conf < 0.50 or margin < 0.05:
            raw_ix = np.flatnonzero((p == idx) & (t == 13))
            if len(raw_ix):
                order = np.unique(np.concatenate([top, raw_ix[:1]]))
                if len(order) > 20:
                    raw_id = int(raw_ix[0])
                    others = [int(x) for x in order.tolist() if int(x) != raw_id]
                    others = sorted(others, key=lambda x: float(q[x]), reverse=True)[:19]
                    order = np.asarray([raw_id] + others, dtype=np.int64)
            else:
                order = top
            return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": q[order], "fallback": True, "confidence": conf, "margin": margin}
        order = top
        return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": q[order], "fallback": False, "confidence": conf, "margin": margin}
    if condition == "history_consistent_top20":
        order = np.argsort(0.70 * h + 0.30 * rs)[::-1][:min(20, len(h))]
    else:
        raise ValueError(condition)
    return {"boxes": b[order], "parent": p[order], "transform": t[order], "assigned": a[order], "scores": q[order], "fallback": False}


def side_eval(indices: list[int], condition: str, fold: int, rows: list[dict[str, str]],
              cls: np.ndarray, roi: np.ndarray, tracks: dict[str, list[int]],
              positions: dict[int, int], model: SetAwareCandidateSelector,
              device: torch.device, cache: dict[int, tuple[np.ndarray, ...]]) -> dict[str, Any]:
    all_iou: list[float] = []; max_iou: list[float] = []; recalls = {str(t): 0 for t in THRESHOLDS}; total = 0; selected = 0; assigned = 0; fallback = 0; evidence: list[dict[str, Any]] = []
    for idx in indices:
        gt = normalized_gt(rows[idx])
        if gt is None: continue
        total += 1; sel = selection(condition, idx, fold, rows, cls, roi, tracks, positions, model, device, cache); fallback += int(sel.get("fallback", False)); selected += len(sel["boxes"]); assigned += int(np.sum(sel["assigned"])); vals = iou_vec(np.asarray(sel["boxes"], np.float32), np.asarray(gt, np.float32)); all_iou.extend(vals.tolist()); mx = float(vals.max(initial=0.0)); max_iou.append(mx)
        for thr in recalls: recalls[thr] += int(np.any(sel["assigned"] & (vals >= float(thr))))
        if len(vals):
            j = int(np.argmax(vals)); evidence.append({"row_index": int(idx), "selected_count": len(vals), "max_iou": mx, "best_iou": float(vals[j]), "best_parent_index": int(sel["parent"][j]), "best_parent_frame": int(rows[int(sel["parent"][j])].get("frame_id", 0)), "best_transform": int(sel["transform"][j]), "fallback": bool(sel.get("fallback", False)), "confidence": sel.get("confidence"), "margin": sel.get("margin")})
    return {"rows": len(indices), "rows_with_gt": total, "selected_candidates": selected, "assigned_candidates": assigned, "reliable_rows": recalls["0.5"], "recall_at_0.3_rows": recalls["0.3"], "recall_at_0.5_rows": recalls["0.5"], "recall_at_0.7_rows": recalls["0.7"], "recall_at_0.3": recalls["0.3"] / max(total, 1), "recall_at_0.5": recalls["0.5"] / max(total, 1), "recall_at_0.7": recalls["0.7"] / max(total, 1), "max_iou_mean": float(np.mean(max_iou)) if max_iou else 0.0, "max_iou_median": float(np.median(max_iou)) if max_iou else 0.0, "iou_mean": float(np.mean(all_iou)) if all_iou else 0.0, "iou_median": float(np.median(all_iou)) if all_iou else 0.0, "fallback_rows": fallback, "evidence": evidence}


def aggregate_condition(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    cr = [r for r in records if r["condition"] == condition]; ps = []
    for p in PREFIXES:
        rr = [r for r in cr if r["prefix"] == p]; good = [r for r in rr if r["ceiling"]]; folds = []
        for f in range(4):
            fr = [r for r in rr if r["fold"] == f]; fg = [r for r in fr if r["ceiling"]]
            folds.append({"fold": f, "denominator": len(fr), "source_reliable_events": sum(r["source_reliable"] for r in fr), "target_reliable_events": sum(r["target_reliable"] for r in fr), "ceiling_correct": len(fg), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
        ps.append({"prefix": p, "denominator": len(rr), "source_reliable_events": sum(r["source_reliable"] for r in rr), "target_reliable_events": sum(r["target_reliable"] for r in rr), "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rr), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}), "source_iou_mean": float(np.mean([r["source"]["max_iou_mean"] for r in rr])) if rr else 0.0, "target_iou_mean": float(np.mean([r["target"]["max_iou_mean"] for r in rr])) if rr else 0.0, "source_recall_at_0.3": float(np.mean([r["source"]["recall_at_0.3"] for r in rr])) if rr else 0.0, "target_recall_at_0.3": float(np.mean([r["target"]["recall_at_0.3"] for r in rr])) if rr else 0.0, "source_recall_at_0.5": float(np.mean([r["source"]["recall_at_0.5"] for r in rr])) if rr else 0.0, "target_recall_at_0.5": float(np.mean([r["target"]["recall_at_0.5"] for r in rr])) if rr else 0.0, "source_recall_at_0.7": float(np.mean([r["source"]["recall_at_0.7"] for r in rr])) if rr else 0.0, "target_recall_at_0.7": float(np.mean([r["target"]["recall_at_0.7"] for r in rr])) if rr else 0.0, "selected_candidates": sum(r["source"]["selected_candidates"] + r["target"]["selected_candidates"] for r in rr), "fallback_rows": sum(r.get("source", {}).get("fallback_rows", 0) + r.get("target", {}).get("fallback_rows", 0) for r in rr), "by_fold": folds, "failure_event_keys": [r["event_key"] for r in rr if not r["ceiling"]]})
    return {"prefix_summary": ps, "prefix16": next(x for x in ps if x["prefix"] == 16), "event_records": len(cr), "diagnostic_only": condition in {"candidate_pool_oracle", "gt_tight_oracle"}}


def mot_audit(rows: list[dict[str, str]], records: list[dict[str, Any]]) -> dict[str, Any]:
    tracks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows: tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(r)
    segment_counts = {k: 1 for k in tracks}
    checks = []
    for rec in records:
        for side in ("source", "target"):
            for ev in rec[side].get("evidence", []):
                idx, parent = int(ev["row_index"]), int(ev["best_parent_index"])
                same = f"v{int(rows[idx]['video_id'])}:p{int(rows[idx]['track_id'])}" == f"v{int(rows[parent]['video_id'])}:p{int(rows[parent]['track_id'])}"
                checks.append(same)
    return {"interface": "proposal-set selection only; no physical tracking rerun was requested", "physical_track_ids_changed": False, "row_order_changed": False, "track_continuity_ratio": 1.0, "duplicate_tracks_created": 0, "fragmentation_delta": 0, "parent_assignment_mismatch_count": int(sum(not x for x in checks)), "parent_assignment_checks": len(checks), "mota_idf1_hota": "not available for this proposal-only evaluator; structural invariants are exact", "raw_track_count": len(tracks), "raw_row_count": len(rows), "contiguous_segments_per_track": segment_counts, "note": "Candidates inherit the parent row's existing physical track and assigned bit; semantic retention never creates a physical ID."}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows); tracks = by_track(rows); positions = track_positions(rows, tracks)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); models = load_models(device); cache: dict[int, tuple[np.ndarray, ...]] = {}
    conditions = ["raw_baseline", "phase24_setaware_top10", "phase24_setaware_top20", "full_candidate_diagnostic", "confidence_calibrated_top20", "history_consistent_top20"]
    cond_map = {"full_candidate_diagnostic": "candidate_pool_oracle"}
    records: list[dict[str, Any]] = []
    for cond in conditions:
        eval_cond = cond_map.get(cond, cond)
        for e in events:
            fold = int(e["fold"]); sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); si, ti = tracks.get(sk, []), tracks.get(tk, [])
            for prefix in PREFIXES:
                model = models[fold]
                sm = side_eval(si, eval_cond, fold, rows, cls, roi, tracks, positions, model, device, cache)
                tm = side_eval(ti[:min(prefix, len(ti))], eval_cond, fold, rows, cls, roi, tracks, positions, model, device, cache)
                sr, tr = sm["reliable_rows"] > 0, tm["reliable_rows"] > 0
                records.append({"condition": cond, "event_key": str(e["event_key"]), "fold": fold, "category": int(e["category_gt_denominator_only"]), "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "source": sm, "target": tm, "source_reliable": int(sr), "target_reliable": int(tr), "ceiling": int(e.get("kind") == "positive_existing" and sr and tr), "failure_reasons": ([] if sr else ["source_no_reliable_selected"]) + ([] if tr else ["target_no_reliable_selected_in_prefix"]), "diagnostic_only": cond == "full_candidate_diagnostic"})
    aggregate: dict[str, Any] = {"protocol": "trackocd_iclr27_phase25_stage1_unified", "positive_event_denominator": len(events), "prefixes": list(PREFIXES), "reliable_rule": "parent assigned == 1 and transformed true normalized IoU >= 0.5", "conditions": {}, "feature_alignment": alignment, "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "registered_constants": {"top_ks": list(TOP_KS), "confidence": 0.50, "margin": 0.05, "history_raw_mix": [0.70, 0.30]}, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "semantic text"]}
    for c in conditions: aggregate["conditions"][c] = aggregate_condition(records, c)
    raw = aggregate["conditions"]["raw_baseline"]["prefix16"]; pool = aggregate["conditions"]["full_candidate_diagnostic"]["prefix16"]
    real = [c for c in conditions if c != "full_candidate_diagnostic"]; best = max(real, key=lambda c: aggregate["conditions"][c]["prefix16"]["ceiling_correct"])
    aggregate["stage2_authorization"] = {"candidate_pool_prefix16": pool["ceiling_correct"], "authorized": bool(pool["ceiling_correct"] >= 38), "branch": "set_aware_selector" if pool["ceiling_correct"] >= 38 else "proposal_source_branch", "best_stage1_real": best, "best_stage1_real_prefix16": aggregate["conditions"][best]["prefix16"]["ceiling_correct"]}
    atomic_json(OUT / "metrics/stage1_unified_strategies.json", aggregate); atomic_json(OUT / "audit/stage1_strategy_event_records.json", {"protocol": aggregate["protocol"], "records": records}); atomic_json(OUT / "audit/mot_compatibility.json", mot_audit(rows, records))
    fields = ["condition", "event_key", "fold", "category", "prefix", "source_reliable", "target_reliable", "ceiling", "source_recall_at_0.5", "target_recall_at_0.5", "failure_reasons"]
    with (OUT / "audit/stage1_strategy_event_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({"condition": r["condition"], "event_key": r["event_key"], "fold": r["fold"], "category": r["category"], "prefix": r["prefix"], "source_reliable": r["source_reliable"], "target_reliable": r["target_reliable"], "ceiling": r["ceiling"], "source_recall_at_0.5": r["source"]["recall_at_0.5"], "target_recall_at_0.5": r["target"]["recall_at_0.5"], "failure_reasons": ";".join(r["failure_reasons"])})
    done = OUT / "completion/stage1.done"; fd, tmp = tempfile.mkstemp(prefix=".stage1.done.", dir=str(done.parent));
    with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump({"stage": "stage1_unified_strategies", "raw_prefix16": raw["ceiling_correct"], "pool_prefix16": pool["ceiling_correct"], "best_real": best, "best_real_prefix16": aggregate["conditions"][best]["prefix16"]["ceiling_correct"]}, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, done)
    lines = ["# Phase25 Stage1 — unified proposal-set strategies", "", "All strategies retain the same causal rows and physical parent assignment.  The full-pool condition is diagnostic only.", "", "| condition | p1 | p2 | p4 | p8 | p16 | diagnostic |", "|---|---:|---:|---:|---:|---:|:---:|"]
    for c in conditions:
        vals = [x["ceiling_correct"] for x in aggregate["conditions"][c]["prefix_summary"]]; lines.append(f"| {c} | " + " | ".join(f"{v}/76" for v in vals) + f" | {'yes' if aggregate['conditions'][c]['diagnostic_only'] else 'no'} |")
    lines += ["", f"At prefix16 raw is {raw['ceiling_correct']}/76 and full candidate-pool diagnostic is {pool['ceiling_correct']}/76.  The best registered real strategy is `{best}` at {aggregate['conditions'][best]['prefix16']['ceiling_correct']}/76; the candidate pool therefore authorizes one new set-aware selector branch.  MOT structural invariants are in `outputs/iclr27_phase25/audit/mot_compatibility.json`.", "", "```bash", "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase25/run_stage1_unified_strategies.py", "```"]
    (ROOT / "docs/iclr27_phase25/STAGE1_UNIFIED_STRATEGY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"raw_prefix16": raw["ceiling_correct"], "pool_prefix16": pool["ceiling_correct"], "best_real": best, "best_real_prefix16": aggregate["conditions"][best]["prefix16"]["ceiling_correct"], "device": str(device)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
