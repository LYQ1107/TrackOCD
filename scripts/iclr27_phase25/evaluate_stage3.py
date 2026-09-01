#!/usr/bin/env python3
"""Phase25 Stage3: frozen attention selector on the exact 76-event protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.iclr27_phase24.protocol import CSV_PATH, FEAT_PATH, PREFIXES, TRANSFORM_META, by_track, candidate_arrays, fval, load_aligned_features, load_events, normalized_gt, raw_box, track_positions
from src.iclr27_phase25.set_selector import ProposalSetAttentionSelector
from scripts.iclr27_phase23.train_quality_ranker import feature_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase25"
TOP_KS = (5, 10, 20, 27)


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


def iou_vec(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1]); x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3]); inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1); aa = np.maximum(0., boxes[:, 2] - boxes[:, 0]) * np.maximum(0., boxes[:, 3] - boxes[:, 1]); ab = max(0., gt[2] - gt[0]) * max(0., gt[3] - gt[1]); return inter / np.maximum(aa + ab - inter, 1e-8)


def load_models(device: torch.device) -> dict[int, ProposalSetAttentionSelector]:
    out = {}
    for f in range(4):
        p = OUT / "checkpoints" / f"attention_f{f}_best.pt"; ck = torch.load(p, map_location="cpu", weights_only=False); m = ProposalSetAttentionSelector(); m.load_state_dict(ck["model"]); m.to(device).eval(); out[f] = m
    return out


def score_row(idx: int, rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, tracks: dict[str, list[int]], positions: dict[int, int], model: ProposalSetAttentionSelector, device: torch.device, cache: dict[int, tuple[np.ndarray, ...]]) -> tuple[np.ndarray, ...]:
    if idx in cache: return cache[idx]
    b, p, t, a = candidate_arrays(rows, idx, tracks, positions); cur = np.full(len(b), idx, np.int32); v, g = feature_batch(rows, cls, roi, cur, p, b, TRANSFORM_META[t])
    with torch.no_grad(): q, u = model(torch.from_numpy(v).to(device).unsqueeze(0), torch.from_numpy(g).to(device).unsqueeze(0), torch.ones((1, len(b)), dtype=torch.bool, device=device))
    out = (b, p, t, a, q.float().cpu().numpy().reshape(-1), u.float().cpu().numpy().reshape(-1)); cache[idx] = out; return out


def side(indices: list[int], model: ProposalSetAttentionSelector, k: int, rows: list[dict[str, str]], cls: np.ndarray, roi: np.ndarray, tracks: dict[str, list[int]], positions: dict[int, int], device: torch.device, cache: dict[int, tuple[np.ndarray, ...]]) -> dict[str, Any]:
    rec = {str(t): 0 for t in (.3, .5, .7)}; all_i: list[float] = []; max_i: list[float] = []; selected = 0; assigned = 0; evidence: list[dict[str, Any]] = []; n = 0
    for idx in indices:
        gt = normalized_gt(rows[idx]);
        if gt is None: continue
        n += 1; b, p, t, a, q, u = score_row(idx, rows, cls, roi, tracks, positions, model, device, cache); order = np.argsort(q)[::-1][:min(k, len(q))]; vals = iou_vec(b[order], np.asarray(gt, np.float32)); aa = a[order]; selected += len(order); assigned += int(aa.sum()); all_i.extend(vals.tolist()); mx = float(vals.max(initial=0.)); max_i.append(mx)
        for thr in rec: rec[thr] += int(np.any(aa & (vals >= float(thr))))
        if len(vals):
            j = int(np.argmax(vals)); evidence.append({"row_index": int(idx), "selected_count": len(order), "max_iou": mx, "best_iou": float(vals[j]), "best_parent_index": int(p[order[j]]), "best_parent_frame": int(rows[int(p[order[j]])].get("frame_id", 0)), "best_transform": int(t[order[j]])})
    return {"rows": len(indices), "rows_with_gt": n, "selected_candidates": selected, "assigned_candidates": assigned, "reliable_rows": rec["0.5"], "recall_at_0.3_rows": rec["0.3"], "recall_at_0.5_rows": rec["0.5"], "recall_at_0.7_rows": rec["0.7"], "recall_at_0.3": rec["0.3"] / max(n, 1), "recall_at_0.5": rec["0.5"] / max(n, 1), "recall_at_0.7": rec["0.7"] / max(n, 1), "max_iou_mean": float(np.mean(max_i)) if max_i else 0., "max_iou_median": float(np.median(max_i)) if max_i else 0., "iou_mean": float(np.mean(all_i)) if all_i else 0., "iou_median": float(np.median(all_i)) if all_i else 0., "evidence": evidence}


def aggregate(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    rr0 = [r for r in records if r["condition"] == condition]; ps = []
    for p in PREFIXES:
        rr = [r for r in rr0 if r["prefix"] == p]; good = [r for r in rr if r["ceiling"]]; folds = []
        for f in range(4):
            fr = [r for r in rr if r["fold"] == f]; fg = [r for r in fr if r["ceiling"]]; folds.append({"fold": f, "denominator": len(fr), "source_reliable_events": sum(r["source_reliable"] for r in fr), "target_reliable_events": sum(r["target_reliable"] for r in fr), "ceiling_correct": len(fg), "category_coverage": len({r["category"] for r in fg}), "video_coverage": len({r["target_video"] for r in fg})})
        ps.append({"prefix": p, "denominator": len(rr), "source_reliable_events": sum(r["source_reliable"] for r in rr), "target_reliable_events": sum(r["target_reliable"] for r in rr), "ceiling_correct": len(good), "ceiling_recall": len(good) / max(len(rr), 1), "category_coverage": len({r["category"] for r in good}), "video_coverage": len({r["target_video"] for r in good}), "source_iou_mean": float(np.mean([r["source"]["max_iou_mean"] for r in rr])) if rr else 0., "target_iou_mean": float(np.mean([r["target"]["max_iou_mean"] for r in rr])) if rr else 0., "source_recall_at_0.3": float(np.mean([r["source"]["recall_at_0.3"] for r in rr])) if rr else 0., "target_recall_at_0.3": float(np.mean([r["target"]["recall_at_0.3"] for r in rr])) if rr else 0., "source_recall_at_0.5": float(np.mean([r["source"]["recall_at_0.5"] for r in rr])) if rr else 0., "target_recall_at_0.5": float(np.mean([r["target"]["recall_at_0.5"] for r in rr])) if rr else 0., "source_recall_at_0.7": float(np.mean([r["source"]["recall_at_0.7"] for r in rr])) if rr else 0., "target_recall_at_0.7": float(np.mean([r["target"]["recall_at_0.7"] for r in rr])) if rr else 0., "by_fold": folds, "failure_event_keys": [r["event_key"] for r in rr if not r["ceiling"]]})
    return {"prefix_summary": ps, "prefix16": next(x for x in ps if x["prefix"] == 16), "event_records": len(rr0), "diagnostic_only": False}


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows); tracks = by_track(rows); positions = track_positions(rows, tracks); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); models = load_models(device); cache: dict[int, tuple[np.ndarray, ...]] = {}; records: list[dict[str, Any]] = []
    for k in TOP_KS:
        cond = f"phase25_attention_top{k}"
        for e in events:
            fold = int(e["fold"]); sk, tk = str(e["source_tracklet_keys"][0]), str(e["target_tracklet_key"]); si, ti = tracks.get(sk, []), tracks.get(tk, []); model = models[fold]
            for prefix in PREFIXES:
                sm, tm = side(si, model, k, rows, cls, roi, tracks, positions, device, cache), side(ti[:min(prefix, len(ti))], model, k, rows, cls, roi, tracks, positions, device, cache); sr, tr = sm["reliable_rows"] > 0, tm["reliable_rows"] > 0; records.append({"condition": cond, "event_key": str(e["event_key"]), "fold": fold, "category": int(e["category_gt_denominator_only"]), "source_video": int(e["source_video"]), "target_video": int(e["target_video"]), "prefix": int(prefix), "source": sm, "target": tm, "source_reliable": int(sr), "target_reliable": int(tr), "ceiling": int(e.get("kind") == "positive_existing" and sr and tr), "failure_reasons": ([] if sr else ["source_no_reliable_selected"]) + ([] if tr else ["target_no_reliable_selected_in_prefix"])})
    stage1 = json.loads((OUT / "metrics/stage1_unified_strategies.json").read_text(encoding="utf-8")); agg: dict[str, Any] = {"protocol": "trackocd_iclr27_phase25_stage3", "positive_event_denominator": len(events), "prefixes": list(PREFIXES), "reliable_rule": "parent assigned == 1 and transformed true normalized IoU >= 0.5", "feature_alignment": alignment, "source_csv": str(CSV_PATH), "source_csv_sha256": sha256(CSV_PATH), "feature_sha256": sha256(FEAT_PATH), "conditions": {}, "model_checkpoints": {f"phase25_attention_top{k}": [str(OUT / "checkpoints" / f"attention_f{f}_best.pt") for f in range(4)] for k in TOP_KS}, "sealed_inputs_not_read": ["DEV+", "Q1", "public new-model labels", "future frames/tracks", "semantic text"]}
    for k in TOP_KS: agg["conditions"][f"phase25_attention_top{k}"] = aggregate(records, f"phase25_attention_top{k}")
    # Inherit all Stage1 conditions without modifying their source artifact.
    for c in ("raw_baseline", "phase24_setaware_top10", "phase24_setaware_top20", "full_candidate_diagnostic", "confidence_calibrated_top20", "history_consistent_top20"):
        agg["conditions"][c] = stage1["conditions"][c]
    raw = agg["conditions"]["raw_baseline"]["prefix16"]; pool = agg["conditions"]["full_candidate_diagnostic"]["prefix16"]; attention = [f"phase25_attention_top{k}" for k in TOP_KS]; best = max(attention, key=lambda c: agg["conditions"][c]["prefix16"]["ceiling_correct"]); b = agg["conditions"][best]["prefix16"]
    fold_dir = sum(x["ceiling_correct"] > y["ceiling_correct"] for x, y in zip(b["by_fold"], raw["by_fold"])); agg["best_real"] = {"condition": best, "prefix16": b["ceiling_correct"]}; agg["gate_p2"] = {"threshold": 38, "raw_prefix16": raw["ceiling_correct"], "candidate_pool_oracle_prefix16": pool["ceiling_correct"], "attention_candidates": {c: agg["conditions"][c]["prefix16"]["ceiling_correct"] for c in attention}, "real_model": best, "source_improved": b["source_reliable_events"] > raw["source_reliable_events"], "target_improved": b["target_reliable_events"] > raw["target_reliable_events"], "folds_improved": fold_dir, "pass": bool(b["ceiling_correct"] >= 38 and b["source_reliable_events"] > raw["source_reliable_events"] and b["target_reliable_events"] > raw["target_reliable_events"] and fold_dir >= 3), "decision": "P25_GATE_P2_PASS" if bool(b["ceiling_correct"] >= 38 and b["source_reliable_events"] > raw["source_reliable_events"] and b["target_reliable_events"] > raw["target_reliable_events"] and fold_dir >= 3) else ("P25_GATE_P2_PARTIAL" if b["ceiling_correct"] >= 30 else "P25_GATE_P2_FAIL")}
    atomic_json(OUT / "metrics/stage3_proposal_validation.json", agg); atomic_json(OUT / "audit/stage3_attention_event_records.json", {"protocol": agg["protocol"], "records": records}); fields = ["condition", "event_key", "fold", "category", "prefix", "source_reliable", "target_reliable", "ceiling", "failure_reasons"]
    with (OUT / "audit/stage3_attention_event_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records: w.writerow({"condition": r["condition"], "event_key": r["event_key"], "fold": r["fold"], "category": r["category"], "prefix": r["prefix"], "source_reliable": r["source_reliable"], "target_reliable": r["target_reliable"], "ceiling": r["ceiling"], "failure_reasons": ";".join(r["failure_reasons"])})
    done = OUT / "completion/stage3.done"; fd, tmp = tempfile.mkstemp(prefix=".stage3.done.", dir=str(done.parent));
    with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump({"stage": "stage3_attention_validation", "best_real": best, "best_prefix16": b["ceiling_correct"], "gate": agg["gate_p2"]["decision"]}, f, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, done)
    lines = ["# Phase25 Stage3 — frozen proposal validation", "", "All real conditions use the original 76 positive events and five causal prefixes.  Candidate-pool and GT-tight values are diagnostics only.", "", "| condition | p1 | p2 | p4 | p8 | p16 |", "|---|---:|---:|---:|---:|---:|"]
    order = ["raw_baseline", "phase24_setaware_top20", "confidence_calibrated_top20", "history_consistent_top20", *attention, "full_candidate_diagnostic"]
    for c in order:
        vals = [x["ceiling_correct"] for x in agg["conditions"][c]["prefix_summary"]]; lines.append(f"| {c} | " + " | ".join(f"{v}/76" for v in vals) + " |")
    lines += ["", f"Best Phase25 attention condition is `{best}` at {b['ceiling_correct']}/76 prefix16; Gate P2 is **{agg['gate_p2']['decision']}**.  Full event records are in `outputs/iclr27_phase25/audit/stage3_attention_event_records.json`.", "", "```bash", "PYTHONPATH=. /home/lwr/anaconda3/envs/locatemot/bin/python scripts/iclr27_phase25/evaluate_stage3.py", "```"]
    (ROOT / "docs/iclr27_phase25/STAGE3_PROPOSAL_VALIDATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"attention": {c: agg["conditions"][c]["prefix16"]["ceiling_correct"] for c in attention}, "best": best, "gate": agg["gate_p2"]["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
