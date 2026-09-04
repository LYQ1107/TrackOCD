#!/usr/bin/env python3
"""Train/replay B4 listwise matcher on the native Q0 candidate universe."""
from __future__ import annotations

import ast
import argparse
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
from scripts.iclr27_phase83.run_b3_joint_support import JointMatcher, group_metrics

OUT = ROOT / "outputs/iclr27_phase83"
DATA = Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_sets/b4_native_sets_v1.npz")
MANIFEST = OUT / "manifests/b4_native_sets_v1.json"
NATIVE = Path("/data2/usr_for_deadline/trackocd_phase83/a2_full/native_lineage.jsonl")
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
OBS = Path("/data2/usr_for_deadline/trackocd_phase75b/observability_repair2/event_observability.jsonl")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def box(v: Any) -> list[float] | None:
    if v is None or v == "": return None
    try: return [float(x) for x in (v if isinstance(v, (list, tuple)) else ast.literal_eval(str(v)))]
    except Exception: return None


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b: return 0.0
    x1,y1,x2,y2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1); aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1]); return inter/max(aa+bb-inter,1e-8)


def replay(models: dict[int, tuple[JointMatcher,np.ndarray,np.ndarray,int]], native: list[dict[str,Any]], x: np.ndarray, flat: np.ndarray, obs: Path, gt_by_key: dict[str,list[float]], groups: dict[tuple[int,int],list[int]]) -> dict[str,Any]:
    native_to_local={int(ni):i for i,ni in enumerate(flat)}; records=[]
    for line in obs.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        e=json.loads(line); fold=int(e["fold"])
        if fold not in models: continue
        model,mean,std,step=models[fold]; xf=(x-mean)/std
        def side(name:str)->dict[str,Any]:
            selected=[]; rel=[]; details=e.get(name+"_row_details",[])
            for d in details:
                nis=groups.get((int(d.get("video_id",-1)),int(d.get("image_id",-1))),[]); local=[native_to_local[n] for n in nis if n in native_to_local]
                if not local: continue
                choice,p=model.choose(xf[np.asarray(local,np.int64)])
                if choice>=len(local): continue
                ni=nis[choice]; selected.append({"native_index":int(ni),"physical_track_id":native[ni].get("physical_track_id"),"probability":float(p[choice]),"image_id":int(d.get("image_id",-1))}); rel.append(iou(native[ni].get("bbox_xyxy"),gt_by_key.get(str(d.get("row_key"))))>=.5)
            return {"candidate_count":len(details),"support_selected":bool(selected),"selected_reliable":bool(any(rel)),"selected_count":len(selected),"selected":selected[:16],"step":step}
        s,t=side("source"),side("target"); records.append({"event_key":e.get("event_key"),"model_event_uid":e.get("model_event_uid"),"fold":fold,"polarity":e.get("polarity"),"prefix":int(e.get("prefix",0)),"source":s,"target":t,"both_support_selected":s["support_selected"] and t["support_selected"],"both_support_reliable":s["selected_reliable"] and t["selected_reliable"],"frozen_both_reliable":bool(e.get("both_reliable")),"frozen_source_reliable":bool(e.get("source_reliable")),"frozen_target_reliable":bool(e.get("target_reliable"))})
    summary=[]
    for p in (1,2,4,8,16):
        pos=[r for r in records if r["prefix"]==p and r["polarity"]=="positive"];neg=[r for r in records if r["prefix"]==p and r["polarity"]=="negative"];summary.append({"prefix":p,"positive_events":len(pos),"negative_events":len(neg),"frozen_both_reliable":sum(r["frozen_both_reliable"] for r in pos),"learned_source_support_selected":sum(r["source"]["support_selected"] for r in pos),"learned_target_support_selected":sum(r["target"]["support_selected"] for r in pos),"learned_both_support_selected":sum(r["both_support_selected"] for r in pos),"learned_both_support_reliable":sum(r["both_support_reliable"] for r in pos),"negative_both_support_selected":sum(r["both_support_selected"] for r in neg),"negative_both_support_reliable":sum(r["both_support_reliable"] for r in neg)})
    return {"schema_version":"trackocd.phase83.b4.native_replay.v1","records":records,"prefix_summary":summary,"positive_denominator":76,"negative_denominator":76,"posthoc_event_labels":True,"public_dev_q1_sealed_accessed":False}


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--folds",default="0,1,2,3");ap.add_argument("--steps",type=int,default=1000);ap.add_argument("--tag",default="b4_formal");args=ap.parse_args();z=np.load(DATA,allow_pickle=False);x=z["features"].astype(np.float32);flat=z["flat_indices"].astype(np.int64);offsets=z["offsets"].astype(np.int64);targets=z["targets"].astype(np.int64);m=json.loads(MANIFEST.read_text(encoding="utf-8"));outcomp=OUT/"completion";outck=Path("/data2/usr_for_deadline/trackocd_phase83/b4_native_checkpoints");outmet=OUT/"metrics";outcomp.mkdir(exist_ok=True,parents=True);outck.mkdir(exist_ok=True,parents=True);outmet.mkdir(exist_ok=True,parents=True);models={};folds_metrics={}
    local_flat=np.arange(len(x),dtype=np.int64)
    for fold in (int(v) for v in args.folds.split(",") if v.strip()):
        marker=outcomp/f"b4_native_{args.tag}_f{fold}.launched";done=outcomp/f"b4_native_{args.tag}_f{fold}.done"
        if done.exists():continue
        if marker.exists():raise RuntimeError(f"unit already launched without done: {marker}")
        atomic_json(marker,{"phase":"Phase83","route":"B4_NATIVE_SET_MATCHER","tag":args.tag,"fold":fold,"pid":os.getpid(),"created_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
        fs=m["folds"][str(fold)];fit=[int(v) for v in fs["fit_groups"]];val=[int(v) for v in fs["validation_groups"]];fit_rows=np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in fit]);mean=x[fit_rows].mean(0);std=x[fit_rows].std(0);std=np.where(std<1e-5,1.0,std).astype(np.float32);xf=(x-mean)/std;relset={g for g in fit if targets[g]<offsets[g+1]-offsets[g]};non=[g for g in fit if g not in relset];rng=np.random.default_rng(8341+fold);model=JointMatcher(x.shape[1],8341+fold);losses=[]
        for step in range(1,args.steps+1):
            pool=list(relset) if relset and non and rng.random()<.5 else (non if non else fit);gi=int(pool[rng.integers(0,len(pool))]);weight=2.0 if gi in relset else 1.0;inds=np.arange(offsets[gi],offsets[gi+1]);losses.append(model.step(xf[inds],int(targets[gi]),weight));
            if step%500==0 or step==args.steps:model.save(outck/f"b4_native_{args.tag}_f{fold}_step{step:06d}.npz",mean,std,step,fold)
        cp=outck/f"b4_native_{args.tag}_f{fold}_step{args.steps:06d}.npz";tm=group_metrics(model,xf,fit,offsets,local_flat,targets);vm=group_metrics(model,xf,val,offsets,local_flat,targets);obj={"phase":"Phase83","route":"B4_NATIVE_SET_MATCHER","tag":args.tag,"fold":fold,"steps":args.steps,"fit_groups":len(fit),"validation_groups":len(val),"fit_metrics":tm,"validation_metrics":vm,"loss_first":losses[0],"loss_last":losses[-1],"checkpoint":str(cp.resolve()),"checkpoint_sha256":sha(cp),"candidate_manifest":str(MANIFEST.resolve()),"candidate_manifest_sha256":sha(MANIFEST),"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"gt_fields_in_feature_tensor":False};atomic_json(outmet/f"b4_native_{args.tag}_f{fold}.json",obj);atomic_json(done,{"status":"DONE","fold":fold,"tag":args.tag,"checkpoint":str(cp.resolve()),"metrics":str((outmet/f"b4_native_{args.tag}_f{fold}.json").resolve())});models[fold]=JointMatcher.load(cp)[:4];folds_metrics[str(fold)]=obj
    native=[]
    with NATIVE.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():native.append(json.loads(line))
    groups:dict[tuple[int,int],list[int]]=defaultdict(list)
    for i,r in enumerate(native):
        if r.get("bbox_xyxy") is not None:groups[(int(r["video_id"]),int(r["image_id"]))].append(i)
    for k in groups:groups[k].sort(key=lambda i:(int(native[i].get("candidate_rank",0)),int(native[i].get("physical_track_id",-1)),i))
    gt_by_key={}
    for r in csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8")):
        b=box(r.get("gt_bbox_xyxy"));
        if b:gt_by_key[str(r.get("row_key"))]=b
    if models:atomic_json(outmet/f"b4_native_replay_{args.tag}.json",replay(models,native,x,flat,OBS,gt_by_key,groups))
    atomic_json(outmet/f"b4_native_aggregate_{args.tag}.json",{"phase":"Phase83","route":"B4_NATIVE_SET_MATCHER","tag":args.tag,"steps":args.steps,"folds":folds_metrics,"replay":str((outmet/f"b4_native_replay_{args.tag}.json").resolve()) if (outmet/f"b4_native_replay_{args.tag}.json").exists() else None,"public_dev_q1_sealed_accessed":False,"controller_run":False});print(json.dumps({"status":"COMPLETE","route":"B4_NATIVE_SET_MATCHER","tag":args.tag,"folds":sorted(models)},indent=2))


if __name__=="__main__":main()
