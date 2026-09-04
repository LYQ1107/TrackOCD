#!/usr/bin/env python3
"""Audit simple causal native candidate selection after B4 failure."""
from __future__ import annotations

import ast
import csv
import datetime as dt
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase83"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
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
    try: return [float(x) for x in (v if isinstance(v, (list, tuple)) else ast.literal_eval(str(v)))]
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]);bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]);return inter/max(aa+bb-inter,1e-8)


def main() -> None:
    rows={str(r["row_key"]):r for r in csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))}; native=[]; groups=defaultdict(list)
    for line in NATIVE.read_text(encoding="utf-8").splitlines():
        if line.strip(): native.append(json.loads(line))
    for i,r in enumerate(native):
        if r.get("bbox_xyxy") is not None:groups[(int(r["video_id"]),int(r["image_id"]))].append(i)
    for k in groups:groups[k].sort(key=lambda i:(int(native[i].get("candidate_rank",0)),int(native[i].get("physical_track_id",-1)),i))
    records=[]
    for line in OBS.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        e=json.loads(line); sides={}
        for side in ("source","target"):
            chosen=[]; oracle=[]; det=e.get(side+"_row_details",[])
            for d in det:
                inds=groups.get((int(d.get("video_id",-1)),int(d.get("image_id",-1))),[]); gt=box(rows.get(str(d.get("row_key")),{}).get("gt_bbox_xyxy"));
                if not inds:continue
                vals=[iou(native[i].get("bbox_xyxy"),gt) for i in inds]; oracle.append(max(vals,default=0.0)); best=max(range(len(inds)),key=lambda j:(float(native[inds[j]].get("base_score",0.0) or 0.0),-int(native[inds[j]].get("candidate_rank",j))));chosen.append(vals[best])
            sides[side]={"rows":len(det),"oracle_reliable":bool(any(v>=.5 for v in oracle)),"score_reliable":bool(any(v>=.5 for v in chosen)),"oracle_max_iou":max(oracle,default=0.0),"score_max_iou":max(chosen,default=0.0),"score_mean_iou":sum(chosen)/max(len(chosen),1)}
        records.append({"event_key":e.get("event_key"),"fold":int(e.get("fold",-1)),"polarity":e.get("polarity"),"prefix":int(e.get("prefix",0)),"source":sides["source"],"target":sides["target"],"oracle_both":sides["source"]["oracle_reliable"] and sides["target"]["oracle_reliable"],"score_both":sides["source"]["score_reliable"] and sides["target"]["score_reliable"]})
    summary=[]
    for p in (1,2,4,8,16):
        pos=[r for r in records if r["prefix"]==p and r["polarity"]=="positive"];neg=[r for r in records if r["prefix"]==p and r["polarity"]=="negative"];summary.append({"prefix":p,"positive":len(pos),"negative":len(neg),"oracle_source":sum(r["source"]["oracle_reliable"] for r in pos),"oracle_target":sum(r["target"]["oracle_reliable"] for r in pos),"oracle_both":sum(r["oracle_both"] for r in pos),"score_source":sum(r["source"]["score_reliable"] for r in pos),"score_target":sum(r["target"]["score_reliable"] for r in pos),"score_both":sum(r["score_both"] for r in pos),"negative_score_both":sum(r["score_both"] for r in neg),"negative_oracle_both":sum(r["oracle_both"] for r in neg)})
    p16=[r for r in records if r["prefix"]==16 and r["polarity"]=="positive"]; out={"schema_version":"trackocd.phase83.b4.native_selection_audit.v1","created_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"strategy":"per-frame highest frozen native base_score, no learned parameters","prefix_summary":summary,"p16_by_fold":{str(f):{"events":sum(r["fold"]==f for r in p16),"oracle_both":sum(r["fold"]==f and r["oracle_both"] for r in p16),"score_both":sum(r["fold"]==f and r["score_both"] for r in p16)} for f in range(4)},"records":records,"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"labels_posthoc_only":True};atomic_json(OUT/"audit/b4_native_selection_audit.json",out);atomic_json(OUT/"status.json",{"phase":"Phase83","route":"B4_NATIVE_SELECTION_AUDIT","status":"COMPLETE","p16":summary[-1],"public_dev_q1_sealed_accessed":False});atomic_json(OUT/"completion/b4_native_selection_audit.done",{"status":"DONE","metrics":str((OUT/"audit/b4_native_selection_audit.json").resolve())});print(json.dumps(summary[-1],indent=2,sort_keys=True))


if __name__=="__main__":main()
