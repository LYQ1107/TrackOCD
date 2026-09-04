#!/usr/bin/env python3
"""B5 diagnostic: rank native target candidates using prior source-track appearance."""
from __future__ import annotations

import ast
import csv
import datetime as dt
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
from src.iclr27_phase75d.protocol import load_frozen_tracks

OUT = ROOT / "outputs/iclr27_phase83"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
FEATURES = Path("/data2/usr_for_deadline/trackocd_phase83/a2_dino_full_r1/merged/native_dinov2.npz")
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box(v: Any) -> list[float] | None:
    try: return [float(x) for x in (v if isinstance(v, (list,tuple)) else ast.literal_eval(str(v)))]
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:return 0.0
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-8)


def main()->None:
    table=load_frozen_tracks();feats=np.load(FEATURES,allow_pickle=False)["features"].astype(np.float32);feats/=np.maximum(np.linalg.norm(feats,axis=1,keepdims=True),1e-8);native=[];groups=defaultdict(list)
    for line in NATIVE.read_text(encoding="utf-8").splitlines():
        if line.strip():native.append(json.loads(line))
    for i,r in enumerate(native):
        if r.get("bbox_xyxy") is not None:groups[(int(r["video_id"]),int(r["image_id"]))].append(i)
    for k in groups:groups[k].sort(key=lambda i:(int(native[i].get("candidate_rank",0)),int(native[i].get("physical_track_id",-1)),i))
    gt={str(r["row_key"]):box(r.get("gt_bbox_xyxy")) for r in csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))}
    recs=[]
    for line in OBS.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        e=json.loads(line); source_key=str(e.get("source_tracklet_key"));
        if source_key not in table.sequences:continue
        source=table.raw_vector(source_key,None);source/=max(float(np.linalg.norm(source)),1e-8);target=[]
        for d in e.get("target_row_details",[]):
            inds=groups.get((int(d.get("video_id",-1)),int(d.get("image_id",-1))),[])
            if not inds:continue
            scores=feats[np.asarray(inds,np.int64)]@source; j=int(np.argmax(scores));ni=inds[j]; target.append({"native_index":int(ni),"cosine":float(scores[j]),"iou":iou(native[ni].get("bbox_xyxy"),gt.get(str(d.get("row_key")))),"q0_reliable":bool(d.get("q0_reliable",False)),"image_id":int(d.get("image_id",-1))})
        recs.append({"event_key":e.get("event_key"),"model_event_uid":e.get("model_event_uid"),"fold":int(e.get("fold",-1)),"polarity":e.get("polarity"),"prefix":int(e.get("prefix",0)),"source_tracklet_key":source_key,"target_rows":len(e.get("target_row_details",[])),"target_selected":target,"target_selected_reliable":bool(any(x["iou"]>=.5 for x in target)),"frozen_target_reliable":bool(e.get("target_reliable",False)),"frozen_source_reliable":bool(e.get("source_reliable",False))})
    summary=[]
    for p in (1,2,4,8,16):
        pos=[r for r in recs if r["prefix"]==p and r["polarity"]=="positive"];neg=[r for r in recs if r["prefix"]==p and r["polarity"]=="negative"];summary.append({"prefix":p,"positive":len(pos),"negative":len(neg),"frozen_source":sum(r["frozen_source_reliable"] for r in pos),"frozen_target":sum(r["frozen_target_reliable"] for r in pos),"cross_support_target":sum(r["target_selected_reliable"] for r in pos),"cross_support_both":sum(r["frozen_source_reliable"] and r["target_selected_reliable"] for r in pos),"negative_cross_support_target":sum(r["target_selected_reliable"] for r in neg)})
    out={"schema_version":"trackocd.phase83.b5.cross_video_support.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"strategy":"prior completed source-track Q0 raw appearance vs native target candidates; per-frame max cosine","prefix_summary":summary,"records":recs,"native_lineage":str(NATIVE.resolve()),"native_features":str(FEATURES.resolve()),"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"event_labels_posthoc_only":True,"controller_run":False};atomic_json(OUT/"audit/b5_cross_video_support.json",out);atomic_json(OUT/"status.json",{"phase":"Phase83","route":"B5_CROSS_VIDEO_SUPPORT_DIAGNOSTIC","status":"COMPLETE","p16":summary[-1],"public_dev_q1_sealed_accessed":False});atomic_json(OUT/"completion/b5_cross_video_support.done",{"status":"DONE","metrics":str((OUT/"audit/b5_cross_video_support.json").resolve())});print(json.dumps(summary[-1],indent=2,sort_keys=True))


if __name__=="__main__":main()
