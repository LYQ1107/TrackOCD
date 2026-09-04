#!/usr/bin/env python3
"""Audit a single source-conditioned, same-space support score before training.

The audit uses only public TRAIN rows and the frozen native Q0/DINO cache.  GT
category is retained outside tensors for post-hoc metric labels.  It does not
touch the 76 held events or any controller.
"""
from __future__ import annotations

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase75d.protocol import PREFIXES, load_frozen_tracks, order_key
from src.iclr27_phase23.protocol import track_key

NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
PUBLIC = ROOT / "data/iclr27_phase19r/sources/public_rows_corrected.csv"
EPISODES = ROOT / "outputs/iclr27_phase30/manifests"
OUT = ROOT / "outputs/iclr27_phase84"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def parse_box(v: Any) -> list[float] | None:
    try:
        x = [float(z) for z in (json.loads(v) if isinstance(v, str) else v)]
        return x if len(x) == 4 else None
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None: return 0.0
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); inter = max(0., x2-x1)*max(0., y2-y1); aa = max(0., a[2]-a[0])*max(0., a[3]-a[1]); bb = max(0., b[2]-b[0])*max(0., b[3]-b[1]); return inter/max(aa+bb-inter, 1e-8)


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32); return v / max(float(np.linalg.norm(v)), 1e-8)


def score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for rec in records:
        cands = rec["candidates"]; pos = set(rec["positives"]); neg = set(rec["negatives"])
        if not cands or not pos or not neg: continue
        scores = np.asarray(rec["scores"], np.float32); raw = np.asarray(rec["raw_scores"], np.float32); order = np.argsort(scores)[::-1]; raw_order = np.argsort(raw)[::-1]; hit = np.asarray([int(cands[int(i)] in pos) for i in order], np.float32); raw_hit = np.asarray([int(cands[int(i)] in pos) for i in raw_order], np.float32); cum = np.cumsum(hit); raw_cum = np.cumsum(raw_hit); ap = float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(len(pos),1)); raw_ap = float(np.sum(raw_cum/(np.arange(len(raw_hit))+1)*raw_hit)/max(len(pos),1)); ps = scores[np.asarray([c in pos for c in cands])]; ns = scores[np.asarray([c in neg for c in cands])]; rps = raw[np.asarray([c in pos for c in cands])]; rns = raw[np.asarray([c in neg for c in cands])]; rows.append({"query_key":rec["query_key"],"category":rec["category"],"video":rec["video"],"r1":float(hit[0]),"r5":float(hit[:5].max(initial=0.0)),"map":ap,"raw_r1":float(raw_hit[0]),"raw_r5":float(raw_hit[:5].max(initial=0.0)),"raw_map":raw_ap,"hard_negative_gap":float(np.max(ps)-np.max(ns)),"raw_hard_negative_gap":float(np.max(rps)-np.max(rns)),"unsafe_flip":bool(raw_hit[0]>0 and hit[0]<=0)})
    return {"queries":len(rows),"r1":float(np.mean([r["r1"] for r in rows])) if rows else 0.,"r5":float(np.mean([r["r5"] for r in rows])) if rows else 0.,"map":float(np.mean([r["map"] for r in rows])) if rows else 0.,"raw_r1":float(np.mean([r["raw_r1"] for r in rows])) if rows else 0.,"raw_r5":float(np.mean([r["raw_r5"] for r in rows])) if rows else 0.,"raw_map":float(np.mean([r["raw_map"] for r in rows])) if rows else 0.,"hard_negative_gap":float(np.mean([r["hard_negative_gap"] for r in rows])) if rows else 0.,"raw_hard_negative_gap":float(np.mean([r["raw_hard_negative_gap"] for r in rows])) if rows else 0.,"unsafe_flip_count":int(sum(r["unsafe_flip"] for r in rows)),"category_count":len(set(r["category"] for r in rows)),"video_count":len(set(r["video"] for r in rows)),"per_query":rows}


def main() -> None:
    table = load_frozen_tracks(); public = list(csv.DictReader(PUBLIC.open(newline="", encoding="utf-8"))); native = [json.loads(line) for line in NATIVE.open(encoding="utf-8") if line.strip()]; feat = np.asarray(np.load(FEATURES, allow_pickle=False)["features"], np.float32)
    if len(native) != len(feat): raise RuntimeError("native/features row count mismatch")
    by_image: dict[tuple[int,int], list[int]] = defaultdict(list)
    for i, row in enumerate(native):
        if parse_box(row.get("bbox_xyxy")) is not None: by_image[(int(row["video_id"]), int(row.get("image_id",-1)))].append(i)
    matched: dict[str, list[tuple[int,int,float]]] = defaultdict(list)
    for row in public:
        pb = parse_box(row.get("bbox_xyxy")); cands = by_image.get((int(row["video_id"]), int(row["image_id"])), [])
        if pb is None or not cands: continue
        best = max(cands, key=lambda j:(iou(pb,parse_box(native[j].get("bbox_xyxy"))),float(native[j].get("base_score",0.) or 0.),-int(native[j].get("candidate_rank") or 0),-j)); sc = iou(pb,parse_box(native[best].get("bbox_xyxy")))
        if sc >= .5: matched[track_key(row)].append((int(row.get("event_rank",0)),best,sc))
    for k in matched: matched[k].sort(key=lambda x:x[0])
    keys = sorted(set(table.metadata) & set(matched)); key_index={k:i for i,k in enumerate(keys)}; vectors=np.zeros((len(PREFIXES),len(keys),768),np.float32); coverage={str(p):0 for p in PREFIXES}; prototype_vectors=np.zeros((3,len(keys),768),np.float32)
    for p_i,p in enumerate(PREFIXES):
        for k in keys:
            seq=matched[k][:p]; arr=np.asarray([feat[j] for _,j,_ in seq],np.float32)
            if len(arr): vectors[p_i,key_index[k]]=norm(arr.mean(0)); coverage[str(p)]+=1
            else: vectors[p_i,key_index[k]]=table.raw_vector(k,p)
            if p == 16 and len(arr):
                chunks=np.array_split(arr, min(3,len(arr)))
                for ci,ch in enumerate(chunks): prototype_vectors[ci,key_index[k]]=norm(ch.mean(0))
    raw_vectors=np.stack([[table.raw_vector(k,p) for k in keys] for p in PREFIXES]).astype(np.float32)
    fold_outputs=[]; availability=[]
    for fold in range(4):
        manifest=json.loads((EPISODES/f"episode_manifest_f{fold}.json").read_text(encoding="utf-8")); fit=[r for r in manifest["records"] if r.get("split")=="fit" and r.get("kind")=="multi_positive_cross_video"]; val=[r for r in manifest["records"] if r.get("split")=="val" and r.get("kind")=="multi_positive_cross_video"]; fit_keys=sorted({str(r["query_track_key"]) for r in fit if str(r.get("query_track_key")) in key_index}); val_keys=sorted({str(r["query_track_key"]) for r in val if str(r.get("query_track_key")) in key_index});
        for split, rows in (("fit",fit),("val",val)):
            support_ok=sum(1 for r in rows if str(r.get("query_track_key")) in key_index and any(str(s) in key_index for s in r.get("support_track_keys",[]))); availability.append({"fold":fold,"split":split,"episodes":len(rows),"query_mapped":sum(str(r.get("query_track_key")) in key_index for r in rows),"support_mapped":support_ok,"positive_support_total":sum(sum(str(s) in key_index for s in r.get("support_track_keys",[])) for r in rows)})
        for p_i,p in enumerate(PREFIXES):
            records=[]
            for r in val:
                q=str(r.get("query_track_key")); ss=[str(s) for s in r.get("support_track_keys",[]) if str(s) in key_index];
                if q not in key_index or not ss: continue
                qv=int(table.metadata[q]["video"]); candidates=[k for k in val_keys if int(table.metadata[k]["video"])!=qv and k not in ss]; positives=[k for k in candidates if int(table.metadata[k]["category"])==int(table.metadata[q]["category"])]; negatives=[k for k in candidates if int(table.metadata[k]["category"])!=int(table.metadata[q]["category"])]
                if not positives or not negatives: continue
                source_protos=[prototype_vectors[c,key_index[s]] for s in ss for c in range(3) if np.linalg.norm(prototype_vectors[c,key_index[s]])>0]
                if not source_protos: source_protos=[vectors[-1,key_index[s]] for s in ss]
                scores=[]; raw_scores=[]
                for c in candidates:
                    tv=vectors[p_i,key_index[c]]; scores.append(float(max(float(tv@sp) for sp in source_protos))); raw_scores.append(float(raw_vectors[p_i,key_index[q]]@raw_vectors[p_i,key_index[c]]))
                records.append({"query_key":q,"category":int(table.metadata[q]["category"]),"video":qv,"candidates":candidates,"positives":positives,"negatives":negatives,"scores":scores,"raw_scores":raw_scores})
            mm=score_records(records); fold_outputs.append({"fold":fold,"prefix":p,"validation_episode_count":len(val),"metrics":{k:v for k,v in mm.items() if k!="per_query"},"support_set_score":"max cosine over source M=3 prototypes","same_native_space":True})
    aggregate={}
    for p in PREFIXES:
        fs=[x["metrics"] for x in fold_outputs if x["prefix"]==p]; aggregate[str(p)]={k:float(np.mean([m[k] for m in fs])) if fs else 0. for k in ("r1","r5","map","raw_r1","raw_r5","raw_map","hard_negative_gap","raw_hard_negative_gap")}; aggregate[str(p)]["queries"]=sum(m["queries"] for m in fs); aggregate[str(p)]["unsafe_flip_count"]=sum(m["unsafe_flip_count"] for m in fs)
    p16=aggregate["16"]; result={"schema_version":"trackocd.phase84.source_conditioned_signal.v1","phase":"Phase84 B84S Stage0","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"native_lineage":str(NATIVE.resolve()),"native_sha256":sha256(NATIVE),"native_features":str(FEATURES.resolve()),"native_features_sha256":sha256(FEATURES),"public_csv":str(PUBLIC.resolve()),"public_csv_sha256":sha256(PUBLIC),"track_count":len(keys),"mapped_track_count_p16":coverage["16"],"mapping_rule":"same video/image and proposal IoU >= 0.5; native corrected DINO only","prefix_coverage":coverage,"fold_availability":availability,"folds":fold_outputs,"aggregate":aggregate,"p16_signal":{"r1_delta":p16["r1"]-p16["raw_r1"],"map_delta":p16["map"]-p16["raw_map"],"hard_gap_delta":p16["hard_negative_gap"]-p16["raw_hard_negative_gap"],"unsafe_flip_count":p16["unsafe_flip_count"],"folds_non_decreasing_both":sum(int(x["metrics"]["r1"]>=x["metrics"]["raw_r1"] and x["metrics"]["map"]>=x["metrics"]["raw_map"]) for x in fold_outputs if x["prefix"]==16)},"training_authorization":{"signal_audit_only":True,"source_completed_prior_video_contract_not_yet_verified":True,"support_tensor_fields":["native_DINO_track_mean","native_DINO_M3_prototypes","target_current_history","candidate_score_rank_age_motion_geometry"],"forbidden_model_inputs":["category","text","semantic_id","physical_id","future","GT","StateMemory","controller_action"],"teacher_labels_posthoc_only":True,"positive_and_hard_defer_groups":"Phase30 TRAIN episodes only; no held events"},"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"held_events_used_for_model_selection":False}
    atomic_json(OUT/"audit/source_conditioned_signal.json",result); atomic_json(OUT/"status.json",{"phase":"Phase84","route":"B84S_SIGNAL_AUDIT","status":"COMPLETE","p16":result["p16_signal"],"public_dev_q1_sealed_accessed":False}); print(json.dumps(result["p16_signal"],indent=2,sort_keys=True))


if __name__ == "__main__": main()
