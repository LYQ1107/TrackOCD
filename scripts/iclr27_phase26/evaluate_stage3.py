#!/usr/bin/env python3
"""Phase26 Stage3: true-IoU proposal-source validation on all 76 events."""
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

from src.iclr27_phase26.protocol import (CSV_PATH, FEAT_PATH, PREFIXES, TOP_KS,
    broad_candidates, by_track, candidate_arrays, load_aligned_features,
    load_events, normalized_gt, raw_box, track_positions, iou_np)
from src.iclr27_phase26.source_generator import ProposalSourceGenerator

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase26"


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
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def geometry(rows, idx):
    keys = ("score", "box_x1_norm", "box_y1_norm", "box_x2_norm", "box_y2_norm", "box_width_norm", "box_height_norm", "box_area_norm", "box_aspect_log", "border_left_norm", "border_top_norm", "border_right_norm", "border_bottom_norm", "causal_prefix_age_norm", "causal_box_stability_iou")
    return np.asarray([[float(rows[int(idx)].get(k, 0.0) or 0.0) for k in keys]], np.float32)


def nms_parent(boxes, parents, scores, iou_thr=0.7):
    keep = []
    for p in np.unique(parents):
        ix = np.flatnonzero(parents == p); order = ix[np.argsort(scores[ix])[::-1]]
        while len(order):
            j = int(order[0]); keep.append(j); order = order[1:]
            if len(order): order = np.asarray([k for k in order.tolist() if float(iou_np(boxes[[j]], boxes[k])[0]) < iou_thr], np.int64)
    return np.asarray(keep, np.int64)


def aggregate(records, condition):
    cr = [r for r in records if r["condition"] == condition]; ps = []
    for prefix in PREFIXES:
        rr = [r for r in cr if r["prefix"] == prefix]; good = [r for r in rr if r["ceiling"]]; folds = []
        for f in range(4):
            fr = [r for r in rr if r["fold"] == f]; fg = [r for r in fr if r["ceiling"]]; folds.append({"fold": f, "denominator": len(fr), "ceiling_correct": len(fg), "source_reliable_events": sum(x["source_reliable"] for x in fr), "target_reliable_events": sum(x["target_reliable"] for x in fr), "category_coverage": len({x["category"] for x in fg}), "video_coverage": len({x["target_video"] for x in fg})})
        ps.append({"prefix": prefix, "denominator": len(rr), "ceiling_correct": len(good), "ceiling_recall": len(good)/max(len(rr),1), "source_reliable_events": sum(x["source_reliable"] for x in rr), "target_reliable_events": sum(x["target_reliable"] for x in rr), "category_coverage": len({x["category"] for x in good}), "video_coverage": len({x["target_video"] for x in good}), "source_iou_mean": float(np.mean([x["source_max_iou"] for x in rr])) if rr else 0., "target_iou_mean": float(np.mean([x["target_max_iou"] for x in rr])) if rr else 0., "source_iou_median": float(np.median([x["source_max_iou"] for x in rr])) if rr else 0., "target_iou_median": float(np.median([x["target_max_iou"] for x in rr])) if rr else 0., "candidate_count_mean": float(np.mean([x["candidate_count"] for x in rr])) if rr else 0., "by_fold": folds, "failure_event_keys": [x["event_key"] for x in rr if not x["ceiling"]]})
    return {"prefix_summary": ps, "prefix16": ps[-1], "event_records": len(cr), "diagnostic_only": condition.endswith("oracle")}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); args = ap.parse_args()
    rows = list(csv.DictReader(CSV_PATH.open(newline="", encoding="utf-8"))); events = load_events(); cls, roi, alignment = load_aligned_features(rows); tracks = by_track(rows); positions = track_positions(rows); device = torch.device(args.device)
    models = {}
    for f in range(4):
        ck = torch.load(OUT / "checkpoints" / f"source_f{f}_best.pt", map_location="cpu", weights_only=False); m = ProposalSourceGenerator(); m.load_state_dict(ck["model"], strict=True); m.to(device).eval(); models[f] = m
    source_cache = {}
    def generated_for_parent(p, fold):
        key = (fold, int(p))
        if key in source_cache: return source_cache[key]
        v = np.concatenate([cls[[p]], roi[[p]]], axis=1).astype(np.float32, copy=False); g = geometry(rows, p); base = g[:, 1:5]
        with torch.no_grad(): boxes, q = models[fold](torch.from_numpy(v).to(device), torch.from_numpy(g).to(device), torch.from_numpy(base).to(device))
        out = (boxes[0].float().cpu().numpy(), torch.sigmoid(q[0]).float().cpu().numpy()); source_cache[key] = out; return out
    def candidates(idx, fold, mode):
        r = rows[idx]
        if mode == "raw":
            return (np.asarray(raw_box(r), np.float32)[None, :], np.asarray([idx], np.int32), np.asarray([-1], np.int16), np.asarray([str(r.get("assigned", "0")) == "1"], bool), np.asarray([float(r.get("score", 0.) or 0.)], np.float32))
        if mode == "fixed": return (*candidate_arrays(rows, idx, tracks, positions), np.asarray([float(rows[int(p)].get("score", 0.) or 0.) for p in candidate_arrays(rows, idx, tracks, positions)[1]], np.float32))
        if mode == "broad": return (*broad_candidates(rows, idx, tracks, positions), np.asarray([float(rows[int(p)].get("score", 0.) or 0.) for p in broad_candidates(rows, idx, tracks, positions)[1]], np.float32))
        # Source replacement: generated proposals from the current/past four
        # causal parents, with the raw current row kept as an immutable fallback.
        inds = tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"]; pos = positions[idx]; hist = inds[max(0, pos-4+1):pos+1]; bb=[]; pp=[]; ss=[]; aa=[]; tt=[]
        for p in hist:
            gb, gq = generated_for_parent(p, fold)
            for j in range(len(gb)): bb.append(gb[j]); pp.append(p); ss.append(float(gq[j])); aa.append(str(rows[p].get("assigned", "0")) == "1"); tt.append(100+j)
        # append current raw candidate before NMS so a source failure cannot
        # erase the inherited physical proposal.
        bb.append(np.asarray(raw_box(r), np.float32)); pp.append(idx); ss.append(float(r.get("score", 0.) or 0.)); aa.append(str(r.get("assigned", "0")) == "1"); tt.append(-1)
        b=np.asarray(bb,np.float32); p=np.asarray(pp,np.int32); s=np.asarray(ss,np.float32); a=np.asarray(aa,bool); t=np.asarray(tt,np.int16); keep=nms_parent(b,p,s,0.7); order=keep[np.argsort(s[keep])[::-1]]
        return b[order], p[order], t[order], a[order], s[order]
    records=[]
    modes = ["raw", "fixed", "broad", "source"]
    names = {"raw":"raw_baseline", "fixed":"phase20_25_fixed_pool", "broad":"phase26_broad_pool_oracle", "source":"phase26_source_branch_topk"}
    for mode in modes:
        for e in events:
            fold=int(e["fold"]); sk,tk=str(e["source_tracklet_keys"][0]),str(e["target_tracklet_key"]); si,ti=tracks.get(sk,[]),tracks.get(tk,[])
            for prefix in PREFIXES:
                side_stats=[]
                for inds in (si, ti[:min(prefix,len(ti))]):
                    allmax=[]; selected=0; src_rel=False
                    for idx in inds:
                        gt=normalized_gt(rows[idx]);
                        if gt is None: continue
                        b,p,t,a,s=candidates(idx,fold,mode); vals=iou_np(b,np.asarray(gt,np.float32)); allmax.append(float(vals.max(initial=0.))); selected += len(b); src_rel = src_rel or bool(np.any(a & (vals>=.5)))
                    side_stats.append((src_rel, float(max(allmax,default=0.)), selected, float(np.mean(allmax) if allmax else 0.)))
                sr,smx,sc,_=side_stats[0]; tr,tmx,tc,_=side_stats[1]; records.append({"condition":names[mode],"event_key":str(e["event_key"]),"fold":fold,"category":int(e["category_gt_denominator_only"]),"source_video":int(e["source_video"]),"target_video":int(e["target_video"]),"prefix":int(prefix),"source_reliable":int(sr),"target_reliable":int(tr),"ceiling":int(e.get("kind")=="positive_existing" and sr and tr),"source_max_iou":smx,"target_max_iou":tmx,"candidate_count":sc+tc,"failure_reasons":([] if sr else ["source_no_reliable_observation"])+([] if tr else ["target_no_reliable_observation_in_prefix"])})
    # Inherit the exact Phase25 attention top27 records as a frozen comparator.
    p25 = json.loads((ROOT / "outputs/iclr27_phase25/audit/stage3_attention_event_records.json").read_text(encoding="utf-8"))["records"]
    for x in p25:
        if x["condition"] != "phase25_attention_top27": continue
        records.append({"condition":"phase25_attention_top27","event_key":x["event_key"],"fold":int(x["fold"]),"category":int(x["category"]),"source_video":int(x["source_video"]),"target_video":int(x["target_video"]),"prefix":int(x["prefix"]),"source_reliable":int(x["source_reliable"]),"target_reliable":int(x["target_reliable"]),"ceiling":int(x["ceiling"]),"source_max_iou":float(x["source"]["max_iou_mean"]),"target_max_iou":float(x["target"]["max_iou_mean"]),"candidate_count":int(x["source"].get("selected_candidates",0)+x["target"].get("selected_candidates",0)),"failure_reasons":x.get("failure_reasons",[])})
    conditions=["raw_baseline","phase25_attention_top27","phase20_25_fixed_pool","phase26_broad_pool_oracle","phase26_source_branch_topk"]; agg={"protocol":"trackocd_iclr27_phase26_stage3_true_iou","positive_event_denominator":76,"prefixes":list(PREFIXES),"reliable_rule":"parent assigned == 1 and transformed IoU >= 0.5","conditions":{c:aggregate(records,c) for c in conditions},"alignment":alignment,"source_csv":str(CSV_PATH),"source_csv_sha256":sha256(CSV_PATH),"feature_sha256":sha256(FEAT_PATH),"model_checkpoints":{f"source_f{f}":str(OUT/"checkpoints"/f"source_f{f}_best.pt") for f in range(4)},"sealed_inputs_not_read":["DEV+","Q1","public new-model labels","future frames/tracks","physical/semantic IDs","semantic text"]}
    raw=agg["conditions"]["raw_baseline"]["prefix16"]; src=agg["conditions"]["phase26_source_branch_topk"]["prefix16"]; pool=agg["conditions"]["phase26_broad_pool_oracle"]["prefix16"]; fold_improved=sum(x["ceiling_correct"]>y["ceiling_correct"] for x,y in zip(src["by_fold"],raw["by_fold"])); agg["gate_p2"]={"threshold":38,"raw_prefix16":raw["ceiling_correct"],"real_source_prefix16":src["ceiling_correct"],"broad_pool_oracle_prefix16":pool["ceiling_correct"],"source_improved":src["source_reliable_events"]>raw["source_reliable_events"],"target_improved":src["target_reliable_events"]>raw["target_reliable_events"],"folds_improved":fold_improved,"pass":bool(src["ceiling_correct"]>=38 and src["source_reliable_events"]>raw["source_reliable_events"] and src["target_reliable_events"]>raw["target_reliable_events"] and fold_improved>=3),"decision":"P26_GATE_P2_PASS" if src["ceiling_correct"]>=38 and src["source_reliable_events"]>raw["source_reliable_events"] and src["target_reliable_events"]>raw["target_reliable_events"] and fold_improved>=3 else ("P26_GATE_P2_PARTIAL" if src["ceiling_correct"]>=30 else "P26_GATE_P2_FAIL")}
    atomic_json(OUT/"metrics/stage3_proposal_validation.json",agg); atomic_json(OUT/"audit/stage3_event_records.json",{"protocol":agg["protocol"],"records":records});
    with (OUT/"audit/stage3_event_summary.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["condition","event_key","fold","category","prefix","source_reliable","target_reliable","ceiling","source_max_iou","target_max_iou","candidate_count","failure_reasons"]; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:r[k] for k in fields} for r in records])
    atomic_json(OUT/"completion/stage3.done",{"stage":"phase26_stage3_true_iou","gate":agg["gate_p2"]["decision"],"source_prefix16":src["ceiling_correct"],"broad_pool_prefix16":pool["ceiling_correct"]})
    lines=["# Phase26 Stage3 — true-IoU proposal-source validation","","All conditions use the immutable 76-event denominator and causal prefixes.  The broad pool is diagnostic; the source branch is the frozen trained model with causal parent candidates, fixed per-parent NMS IoU=0.7, top-K ranking and raw fallback.","","| condition | p1 | p2 | p4 | p8 | p16 | source@16 | target@16 | categories | videos |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in conditions:
        d=agg["conditions"][c]; vals=[x["ceiling_correct"] for x in d["prefix_summary"]]; p=d["prefix16"]; lines.append(f"| {c} | "+" | ".join(f"{v}/76" for v in vals)+f" | {p['source_reliable_events']} | {p['target_reliable_events']} | {p['category_coverage']} | {p['video_coverage']} |")
    lines += ["",f"Gate P2 is **{agg['gate_p2']['decision']}**: raw {raw['ceiling_correct']}/76, real source branch {src['ceiling_correct']}/76, broad candidate oracle {pool['ceiling_correct']}/76.  Full event records are in `outputs/iclr27_phase26/audit/stage3_event_records.json`."]
    (ROOT/"docs/iclr27_phase26/STAGE3_PROPOSAL_SOURCE_VALIDATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"conditions":{c:agg["conditions"][c]["prefix16"]["ceiling_correct"] for c in conditions},"gate":agg["gate_p2"]},indent=2,sort_keys=True))


if __name__ == "__main__": main()
