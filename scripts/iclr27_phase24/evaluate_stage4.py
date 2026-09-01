#!/usr/bin/env python3
"""Phase24 Stage4 true-IoU validation with frozen set-aware selectors."""
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

from src.iclr27_phase24.protocol import CSV_PATH, FEAT_PATH, PREFIXES, PROTOCOL, RELIABLE_RULE, TRANSFORM_META, by_track, candidate_arrays, fval, load_aligned_features, load_events, normalized_gt, raw_box, track_positions
from src.iclr27_phase24.set_selector import SetAwareCandidateSelector
from scripts.iclr27_phase23.train_quality_ranker import feature_batch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase24"
TOP_KS = (1, 5, 10, 20)


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
    x1 = np.maximum(boxes[:, 0], gt[0]); y1 = np.maximum(boxes[:, 1], gt[1]); x2 = np.minimum(boxes[:, 2], gt[2]); y2 = np.minimum(boxes[:, 3], gt[3]); inter = np.maximum(0., x2 - x1) * np.maximum(0., y2 - y1); aa = np.maximum(0., boxes[:, 2] - boxes[:, 0]); aa = aa * np.maximum(0., boxes[:, 3] - boxes[:, 1]); ab = max(0., gt[2] - gt[0]) * max(0., gt[3] - gt[1]); return inter / np.maximum(aa + ab - inter, 1e-8)


def main() -> None:
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows); tracks = by_track(rows); positions = track_positions(rows, tracks); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    rankers: dict[int, SetAwareCandidateSelector] = {}
    for fold in range(4):
        ckpt = OUT / "checkpoints" / f"setaware_f{fold}_best.pt"; ck = torch.load(ckpt, map_location="cpu", weights_only=False); m = SetAwareCandidateSelector(); m.load_state_dict(ck["model"]); m.to(device).eval(); rankers[fold] = m
    cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}; score_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    def scores(idx: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        key = (fold, idx)
        if key in score_cache: return score_cache[key]
        if idx not in cache: cache[idx] = candidate_arrays(rows, idx, tracks, positions)
        b, p, t, a = cache[idx]; cur = np.full(len(b), idx, dtype=np.int32); v, g = feature_batch(rows, cls, roi, cur, p, b, TRANSFORM_META[t])
        with torch.no_grad(): q, u = rankers[fold](torch.from_numpy(v).to(device).unsqueeze(0), torch.from_numpy(g).to(device).unsqueeze(0), torch.ones((1, len(b)), dtype=torch.bool, device=device));
        out = (b, p, t, a, q.float().cpu().numpy().reshape(-1), u.float().cpu().numpy().reshape(-1)); score_cache[key] = out; return out

    def side(indices: list[int], fold: int, k: int) -> dict[str, Any]:
        max_i, all_i = [], []; recalls = {str(t): 0 for t in (.3, .5, .7)}; rows_gt = 0; selected = 0; assigned_count = 0; evidence=[]
        for idx in indices:
            gt = normalized_gt(rows[idx]);
            if gt is None: continue
            rows_gt += 1; b,p,t,a,q,u = scores(idx, fold); order = np.argsort(q)[::-1][:min(k, len(q))]; b, a, p, t = b[order], a[order], p[order], t[order]; selected += len(b); assigned_count += int(a.sum()); vals = iou_vec(b, np.asarray(gt, np.float32)); mx = float(vals.max(initial=0.)); max_i.append(mx); all_i.extend(vals.tolist())
            for thr in recalls: recalls[thr] += int(np.any(a & (vals >= float(thr))))
            if len(vals):
                j = int(np.argmax(vals)); evidence.append({"row_index": int(idx), "selected_count": len(vals), "max_iou": mx, "best_parent_index": int(p[j]), "best_parent_frame": int(rows[int(p[j])].get("frame_id", 0)), "best_transform": int(t[j])})
        return {"rows": len(indices), "rows_with_gt": rows_gt, "selected_candidates": selected, "assigned_candidates": assigned_count, "reliable_rows": recalls["0.5"], "recall_at_0.3": recalls["0.3"] / max(rows_gt,1), "recall_at_0.5": recalls["0.5"] / max(rows_gt,1), "recall_at_0.7": recalls["0.7"] / max(rows_gt,1), "recall_at_0.3_rows": recalls["0.3"], "recall_at_0.5_rows": recalls["0.5"], "recall_at_0.7_rows": recalls["0.7"], "max_iou_mean": float(np.mean(max_i)) if max_i else 0., "max_iou_median": float(np.median(max_i)) if max_i else 0., "iou_mean": float(np.mean(all_i)) if all_i else 0., "iou_median": float(np.median(all_i)) if all_i else 0., "evidence": evidence}

    records=[]
    for k in TOP_KS:
        cond=f"setaware_top{k}"
        for e in events:
            fold=int(e["fold"]); sk,tk=str(e["source_tracklet_keys"][0]),str(e["target_tracklet_key"]); si,ti=tracks.get(sk,[]),tracks.get(tk,[])
            for prefix in PREFIXES:
                sm,tm=side(si,fold,k),side(ti[:min(prefix,len(ti))],fold,k); sr, tr = sm["reliable_rows"]>0, tm["reliable_rows"]>0
                records.append({"condition":cond,"event_key":str(e["event_key"]),"fold":fold,"category":int(e["category_gt_denominator_only"]),"source_video":int(e["source_video"]),"target_video":int(e["target_video"]),"prefix":int(prefix),"source":sm,"target":tm,"source_reliable":int(sr),"target_reliable":int(tr),"ceiling":int(e.get("kind")=="positive_existing" and sr and tr),"failure_reasons":([] if sr else ["source_no_reliable_selected"])+([] if tr else ["target_no_reliable_selected_in_prefix"]),"diagnostic_only":False})
    agg={"protocol":PROTOCOL+"_stage4_setaware","positive_event_denominator":len(events),"prefixes":list(PREFIXES),"reliable_rule":RELIABLE_RULE,"feature_alignment":alignment,"source_csv":str(CSV_PATH),"source_csv_sha256":sha256(CSV_PATH),"feature_sha256":sha256(FEAT_PATH),"conditions":{},"model_checkpoints":{},"sealed_inputs_not_read":["DEV+","Q1","public new-model labels","future frames/tracks","semantic text"]}
    for k in TOP_KS:
        cond=f"setaware_top{k}"; cr=[r for r in records if r["condition"]==cond]; ps=[]
        for p in PREFIXES:
            rr=[r for r in cr if r["prefix"]==p]; good=[r for r in rr if r["ceiling"]]; by_fold=[]
            for f in range(4):
                fr=[r for r in rr if r["fold"]==f]; fg=[r for r in fr if r["ceiling"]]; by_fold.append({"fold":f,"denominator":len(fr),"source_reliable_events":sum(r["source_reliable"] for r in fr),"target_reliable_events":sum(r["target_reliable"] for r in fr),"ceiling_correct":len(fg),"category_coverage":len({r["category"] for r in fg}),"video_coverage":len({r["target_video"] for r in fg})})
            ps.append({"prefix":p,"denominator":len(rr),"source_reliable_events":sum(r["source_reliable"] for r in rr),"target_reliable_events":sum(r["target_reliable"] for r in rr),"ceiling_correct":len(good),"ceiling_recall":len(good)/max(len(rr),1),"category_coverage":len({r["category"] for r in good}),"video_coverage":len({r["target_video"] for r in good}),"source_iou_mean":float(np.mean([r["source"]["max_iou_mean"] for r in rr])) if rr else 0.,"target_iou_mean":float(np.mean([r["target"]["max_iou_mean"] for r in rr])) if rr else 0.,"source_recall_at_0.5":float(np.mean([r["source"]["recall_at_0.5"] for r in rr])) if rr else 0.,"target_recall_at_0.5":float(np.mean([r["target"]["recall_at_0.5"] for r in rr])) if rr else 0.,"by_fold":by_fold,"failure_event_keys":[r["event_key"] for r in rr if not r["ceiling"]]})
        agg["conditions"][cond]={"prefix_summary":ps,"prefix16":next(x for x in ps if x["prefix"]==16),"event_records":len(cr),"diagnostic_only":False}; agg["model_checkpoints"][cond]=[str(OUT/"checkpoints"/f"setaware_f{f}_best.pt") for f in range(4)]
    # Merge the already-computed no-training conditions without recomputing or changing them.
    stage1=json.loads((OUT/"metrics/stage1_unified_strategies.json").read_text())
    for cond in ("raw_baseline","fixed_combo","phase23_mlp_top1","phase23_mlp_top5","phase23_mlp_top10","phase23_mlp_top20","uncertainty_defer","candidate_pool_oracle","gt_tight_oracle"):
        agg["conditions"][cond]=stage1["conditions"][cond]
    raw=agg["conditions"]["raw_baseline"]["prefix16"]; candidates=[f"setaware_top{k}" for k in TOP_KS]; best=max(candidates,key=lambda c:agg["conditions"][c]["prefix16"]["ceiling_correct"]); b=agg["conditions"][best]["prefix16"]
    agg["stage1_reference"]={"path":str(OUT/"metrics/stage1_unified_strategies.json"),"sha256":sha256(OUT/"metrics/stage1_unified_strategies.json")}; agg["best_setaware"]={"condition":best,"prefix16":b["ceiling_correct"]}; agg["gate_p2"]={"threshold":38,"raw_prefix16":raw["ceiling_correct"],"candidate_pool_oracle_prefix16":agg["conditions"]["candidate_pool_oracle"]["prefix16"]["ceiling_correct"],"setaware_candidates":{c:agg["conditions"][c]["prefix16"]["ceiling_correct"] for c in candidates},"real_model":best,"pass":bool(b["ceiling_correct"]>=38 and b["source_reliable_events"]>raw["source_reliable_events"] and b["target_reliable_events"]>raw["target_reliable_events"] and sum(x["ceiling_correct"]>y["ceiling_correct"] for x,y in zip(b["by_fold"],raw["by_fold"]))>=3),"decision":"P24_GATE_P2_PASS" if bool(b["ceiling_correct"]>=38 and b["source_reliable_events"]>raw["source_reliable_events"] and b["target_reliable_events"]>raw["target_reliable_events"] and sum(x["ceiling_correct"]>y["ceiling_correct"] for x,y in zip(b["by_fold"],raw["by_fold"]))>=3) else ("P24_GATE_P2_PARTIAL" if b["ceiling_correct"]>=30 else "P24_GATE_P2_FAIL")}
    atomic_json(OUT/"metrics/stage4_proposal_validation.json",agg); atomic_json(OUT/"audit/stage4_setaware_event_records.json",{"protocol":agg["protocol"],"records":records})
    fields=["condition","event_key","fold","category","prefix","source_reliable","target_reliable","ceiling","failure_reasons"]
    with (OUT/"audit/stage4_setaware_event_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in records: w.writerow({"condition":r["condition"],"event_key":r["event_key"],"fold":r["fold"],"category":r["category"],"prefix":r["prefix"],"source_reliable":r["source_reliable"],"target_reliable":r["target_reliable"],"ceiling":r["ceiling"],"failure_reasons":";".join(r["failure_reasons"])})
    atomic_json(OUT/"completion/stage4.done",{"stage":"stage4_true_iou_validation","best_setaware":best,"best_prefix16":b["ceiling_correct"],"gate":agg["gate_p2"]["decision"]})
    lines=["# Phase24 Stage4 — frozen set-aware selector validation","","The selector is frozen after TRAIN-only disjoint validation; this report uses the complete 76-event, five-prefix true-IoU evaluator.  Raw/fixed/MLP/oracle conditions are inherited byte-for-byte from Stage1.","","| condition | p1 | p2 | p4 | p8 | p16 |","|---|---:|---:|---:|---:|---:|"]
    for c in ["raw_baseline","fixed_combo","phase23_mlp_top1","phase23_mlp_top5","phase23_mlp_top10","phase23_mlp_top20","uncertainty_defer","setaware_top1","setaware_top5","setaware_top10","setaware_top20","candidate_pool_oracle","gt_tight_oracle"]:
        ps=agg["conditions"][c]["prefix_summary"]; vals=[next(x["ceiling_correct"] for x in ps if x["prefix"]==p) for p in PREFIXES]; lines.append(f"| {c} | " + " | ".join(f"{v}/76" for v in vals) + " |")
    lines += ["",f"Best set-aware condition: `{best}` at {b['ceiling_correct']}/76 prefix16; Gate P2 decision is **{agg['gate_p2']['decision']}**.  Event records are in `outputs/iclr27_phase24/audit/stage4_setaware_event_records.json`; Stage1 reference records remain in `outputs/iclr27_phase24/audit/stage1_strategy_event_records.json`."]
    (ROOT/"docs/iclr27_phase24/STAGE4_VALIDATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"setaware":{c:agg["conditions"][c]["prefix16"]["ceiling_correct"] for c in candidates},"best":best,"gate":agg["gate_p2"]["decision"]},indent=2,sort_keys=True))


if __name__ == "__main__": main()
