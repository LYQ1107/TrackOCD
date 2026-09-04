#!/usr/bin/env python3
"""Phase83 B3: joint support-set matcher after B2 listwise failure.

This route keeps the same per-image candidate competition and explicit DEFER
action, but adds full corrected-DINO candidate-to-history and
candidate-to-set similarities.  All similarities are built from current and
strictly earlier rows; TRAIN labels are used only for the listwise target.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.iclr27_phase23.protocol import load_aligned_features, order_key
from scripts.iclr27_phase83.build_support_candidate_sets import row_features

OUT = ROOT / "outputs/iclr27_phase83"
CSV_PATH = ROOT / "outputs/iclr27_phase17r/csv/public_rows_corrected.csv"
MANIFEST = OUT / "manifests/b2_candidate_sets_v1.json"
DATA_PATH = Path("/data2/usr_for_deadline/trackocd_phase83/b2_candidate_sets/b2_candidate_sets_v1.npz")
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


def norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32); return v / max(float(np.linalg.norm(v)), 1e-8)


class JointMatcher:
    def __init__(self, d: int, seed: int = 8313) -> None:
        rng = np.random.default_rng(seed); self.d = d; self.wc = (rng.standard_normal(d, dtype=np.float32) * np.float32(math.sqrt(2.0 / d))).astype(np.float32); self.bc = np.float32(0.0); self.wd = (rng.standard_normal(2*d, dtype=np.float32) * np.float32(math.sqrt(2.0 / (2*d)))).astype(np.float32); self.bd = np.float32(0.0); self.m = {"wc": np.zeros_like(self.wc), "bc": np.zeros(1, np.float32), "wd": np.zeros_like(self.wd), "bd": np.zeros(1, np.float32)}; self.v = {k: np.zeros_like(x) for k, x in self.m.items()}; self.t = 0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cmean = x.mean(axis=0); cmax = x.max(axis=0); ctx = np.concatenate([cmean, cmax]); logits = np.concatenate([(x @ self.wc + self.bc).astype(np.float32), np.asarray([ctx @ self.wd + self.bd], np.float32)]); logits -= logits.max(); p = np.exp(np.clip(logits, -30, 30)).astype(np.float32); p /= max(float(p.sum()), 1e-8); return p, cmean, cmax

    def step(self, x: np.ndarray, target: int, weight: float, lr: float = .003) -> float:
        p, cmean, cmax = self.forward(x); n = len(x); g = p.copy(); g[target] -= np.float32(1.0); g *= np.float32(weight); ctx = np.concatenate([cmean, cmax]); grads = {"wc": x.T @ g[:n], "bc": np.asarray([g[:n].sum()], np.float32), "wd": ctx * g[n], "bd": np.asarray([g[n]], np.float32)}; self.t += 1
        for k, grad in grads.items():
            self.m[k] = .9*self.m[k] + .1*grad; self.v[k] = .999*self.v[k] + .001*(grad*grad); mh = self.m[k]/(1-.9**self.t); vh = self.v[k]/(1-.999**self.t); upd = np.float32(lr)*mh/(np.sqrt(vh)+1e-8)
            if k == "wc": self.wc -= upd
            elif k == "bc": self.bc = np.float32(self.bc-upd[0])
            elif k == "wd": self.wd -= upd
            else: self.bd = np.float32(self.bd-upd[0])
        return float(-weight*math.log(max(float(p[target]), 1e-8)))

    def choose(self, x: np.ndarray) -> tuple[int, np.ndarray]: return int(np.argmax(self.forward(x)[0])), self.forward(x)[0]

    def save(self, path: Path, mean: np.ndarray, std: np.ndarray, step: int, fold: int) -> None:
        arrays = {"wc":self.wc,"bc":np.asarray([self.bc],np.float32),"wd":self.wd,"bd":np.asarray([self.bd],np.float32),"mean":mean.astype(np.float32),"std":std.astype(np.float32),"step":np.asarray([step],np.int64),"fold":np.asarray([fold],np.int64),"t":np.asarray([self.t],np.int64)}
        for k in self.m: arrays[f"m_{k}"]=self.m[k]; arrays[f"v_{k}"]=self.v[k]
        path.parent.mkdir(parents=True, exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".npz",dir=str(path.parent)); os.close(fd)
        try:
            with open(tmp,"wb") as f: np.savez(f,**arrays); f.flush(); os.fsync(f.fileno())
            os.replace(tmp,path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    @classmethod
    def load(cls, path: Path) -> tuple["JointMatcher",np.ndarray,np.ndarray,int,int]:
        z=np.load(path,allow_pickle=False); o=cls(int(z["wc"].shape[0]),0); o.wc=z["wc"]; o.bc=float(z["bc"][0]); o.wd=z["wd"]; o.bd=float(z["bd"][0]); o.t=int(z.get("t",np.asarray([0]))[0]);
        for k in o.m:
            if f"m_{k}" in z:o.m[k]=z[f"m_{k}"]
            if f"v_{k}" in z:o.v[k]=z[f"v_{k}"]
        return o,z["mean"],z["std"],int(z["step"][0]),int(z["fold"][0])


def make_features(rows: list[dict[str,str]], fused: np.ndarray) -> tuple[np.ndarray,list[str]]:
    base=row_features(rows,fused); dino=np.asarray(fused,np.float32); extra=np.zeros((len(rows),4),np.float32); tracks:dict[str,list[int]]=defaultdict(list); groups:dict[tuple[int,int],list[int]]=defaultdict(list)
    for i,r in enumerate(rows): tracks[f"v{int(r['video_id'])}:p{int(r['track_id'])}"].append(i); groups[(int(r['video_id']),int(r['image_id']))].append(i)
    for inds in tracks.values():
        inds.sort(key=lambda i:order_key(rows[i])); hist=[]
        for i in inds:
            cur=norm(dino[i]);
            if hist:
                sims=np.asarray([float(cur@h) for h in hist],np.float32); extra[i,0]=float(sims.max()); extra[i,1]=float(sims.mean())
            hist.append(cur)
    for inds in groups.values():
        inds.sort(key=lambda i:(int(float(rows[i].get('proposal_local_id',0) or 0)),int(float(rows[i].get('track_id',0) or 0)),i)); mean=norm(dino[inds].mean(axis=0)); sim=dino[inds]@mean; extra[inds,2]=sim; 
        if len(inds)>1:
            mat=dino[inds]@dino[inds].T; np.fill_diagonal(mat,-1); extra[inds,3]=mat.max(axis=1)
    return np.concatenate([base,extra],axis=1), ["history_max_cosine","history_mean_cosine","candidate_set_mean_cosine","candidate_set_nearest_cosine"]


def group_metrics(model:JointMatcher,x:np.ndarray,groups:list[int],offsets:np.ndarray,flat:np.ndarray,targets:np.ndarray)->dict[str,Any]:
    total=len(groups); rel=0; correct=0; predcand=0; defer=0; defer_correct=0; top={1:0,5:0,10:0,20:0}; nll=[]
    for g in groups:
        inds=flat[offsets[g]:offsets[g+1]]; t=int(targets[g]); p,_,_=model.forward(x[inds]); nll.append(-math.log(max(float(p[t]),1e-8))); rel+=int(t<len(inds)); choice=int(np.argmax(p)); predcand+=int(choice<len(inds)); defer+=int(choice>=len(inds)); correct+=int(choice==t); defer_correct+=int(t>=len(inds) and choice>=len(inds));
        if t<len(inds):
            order=np.argsort(-p[:len(inds)])
            for k in top: top[k]+=int(t in order[:k])
    return {"groups":total,"reliable_target_groups":rel,"defer_target_groups":total-rel,"predicted_candidate_groups":predcand,"predicted_defer_groups":defer,"candidate_or_defer_accuracy":correct/max(total,1),"defer_recall":defer_correct/max(total-rel,1),"candidate_topk_recall":{str(k):top[k]/max(rel,1) for k in top},"mean_nll":float(np.mean(nll)) if nll else 0.0}


def replay(models:dict[int,tuple[JointMatcher,np.ndarray,np.ndarray,int]],rows:list[dict[str,str]],x:np.ndarray,groups:dict[tuple[int,int],list[int]])->dict[str,Any]:
    records=[]
    for line in OBS.read_text(encoding="utf-8").splitlines():
        if not line.strip():continue
        e=json.loads(line); fold=int(e["fold"])
        if fold not in models:continue
        model,mean,std,step=models[fold]; xf=(x-mean)/std
        def side(name:str)->dict[str,Any]:
            selected=[]; rel=[]; details=e.get(name+"_row_details",[])
            for d in details:
                inds=groups.get((int(d.get("video_id",-1)),int(d.get("image_id",-1))),[])
                if not inds:continue
                choice,p=model.choose(xf[np.asarray(inds,np.int64)])
                if choice>=len(inds):continue
                idx=inds[choice]; selected.append({"row_key":rows[idx].get("row_key"),"probability":float(p[choice]),"image_id":int(d.get("image_id",-1))}); rel.append(int(float(rows[idx].get("assigned",0) or 0))==1 and float(rows[idx].get("row_iou",0) or 0)>=.5)
            return {"candidate_count":len(details),"support_selected":bool(selected),"selected_reliable":bool(any(rel)),"selected_count":len(selected),"selected":selected[:16],"step":step}
        s,t=side("source"),side("target"); records.append({"event_key":e.get("event_key"),"model_event_uid":e.get("model_event_uid"),"fold":fold,"polarity":e.get("polarity"),"prefix":int(e.get("prefix",0)),"source":s,"target":t,"both_support_selected":s["support_selected"] and t["support_selected"],"both_support_reliable":s["selected_reliable"] and t["selected_reliable"],"frozen_both_reliable":bool(e.get("both_reliable")),"frozen_source_reliable":bool(e.get("source_reliable")),"frozen_target_reliable":bool(e.get("target_reliable"))})
    summary=[]
    for p in (1,2,4,8,16):
        pos=[r for r in records if r["prefix"]==p and r["polarity"]=="positive"]; neg=[r for r in records if r["prefix"]==p and r["polarity"]=="negative"]; summary.append({"prefix":p,"positive_events":len(pos),"negative_events":len(neg),"frozen_both_reliable":sum(r["frozen_both_reliable"] for r in pos),"learned_source_support_selected":sum(r["source"]["support_selected"] for r in pos),"learned_target_support_selected":sum(r["target"]["support_selected"] for r in pos),"learned_both_support_selected":sum(r["both_support_selected"] for r in pos),"learned_both_support_reliable":sum(r["both_support_reliable"] for r in pos),"negative_both_support_selected":sum(r["both_support_selected"] for r in neg),"negative_both_support_reliable":sum(r["both_support_reliable"] for r in neg)})
    return {"schema_version":"trackocd.phase83.b3.joint_support_replay.v1","records":records,"prefix_summary":summary,"positive_denominator":76,"negative_denominator":76,"posthoc_event_labels":True,"public_dev_q1_sealed_accessed":False}


def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--folds",default="0,1,2,3");ap.add_argument("--steps",type=int,default=1000);ap.add_argument("--tag",default="b3_formal");args=ap.parse_args(); z=np.load(DATA_PATH,allow_pickle=False); flat=z["flat_indices"].astype(np.int64); offsets=z["offsets"].astype(np.int64); targets=z["targets"].astype(np.int64); manifest=json.loads(MANIFEST.read_text(encoding="utf-8")); rows=list(csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))); cls,roi,_=load_aligned_features(rows); fused=(.8*cls.astype(np.float32)+.2*roi.astype(np.float32)).astype(np.float32); fused/=np.maximum(np.linalg.norm(fused,axis=1,keepdims=True),1e-8); x,extra_names=make_features(rows,fused); outcomp=OUT/"completion";outck=Path("/data2/usr_for_deadline/trackocd_phase83/b3_joint_checkpoints");outmet=OUT/"metrics";outcomp.mkdir(exist_ok=True,parents=True);outmet.mkdir(exist_ok=True,parents=True);outck.mkdir(exist_ok=True,parents=True); models={}; folds_metrics={}
    for fold in (int(v) for v in args.folds.split(",") if v.strip()):
        marker=outcomp/f"b3_joint_{args.tag}_f{fold}.launched"; done=outcomp/f"b3_joint_{args.tag}_f{fold}.done"
        if done.exists():continue
        if marker.exists():raise RuntimeError(f"unit already launched without done: {marker}")
        atomic_json(marker,{"phase":"Phase83","route":"B3_JOINT_SUPPORT","tag":args.tag,"fold":fold,"pid":os.getpid(),"created_utc":dt.datetime.now(dt.timezone.utc).isoformat()})
        fs=manifest["folds"][str(fold)]; fit=[int(v) for v in fs["fit_groups"]]; val=[int(v) for v in fs["validation_groups"]]; fit_rows=flat[np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in fit])]; mean=x[fit_rows].mean(0);std=x[fit_rows].std(0);std=np.where(std<1e-5,1.0,std).astype(np.float32);xf=(x-mean)/std; relset={g for g in fit if targets[g]<offsets[g+1]-offsets[g]}; non=[g for g in fit if g not in relset]; rng=np.random.default_rng(8313+fold); model=JointMatcher(x.shape[1],8313+fold); losses=[]
        for step in range(1,args.steps+1):
            if relset and non:
                pool=list(relset) if rng.random()<.5 else non
            else:
                pool=fit
            gi=int(pool[rng.integers(0,len(pool))]); weight=2.0 if gi in relset else 1.0; inds=flat[offsets[gi]:offsets[gi+1]];losses.append(model.step(xf[inds],int(targets[gi]),weight));
            if step%500==0 or step==args.steps:model.save(outck/f"b3_joint_{args.tag}_f{fold}_step{step:06d}.npz",mean,std,step,fold)
        cp=outck/f"b3_joint_{args.tag}_f{fold}_step{args.steps:06d}.npz";tm=group_metrics(model,xf,fit,offsets,flat,targets);vm=group_metrics(model,xf,val,offsets,flat,targets);obj={"phase":"Phase83","route":"B3_JOINT_SUPPORT","tag":args.tag,"fold":fold,"steps":args.steps,"fit_groups":len(fit),"validation_groups":len(val),"fit_metrics":tm,"validation_metrics":vm,"loss_first":losses[0],"loss_last":losses[-1],"checkpoint":str(cp.resolve()),"checkpoint_sha256":sha(cp),"feature_names":[f"b2_{i}" for i in range(17)]+extra_names,"candidate_manifest":str(MANIFEST.resolve()),"candidate_manifest_sha256":sha(MANIFEST),"public_dev_q1_sealed_accessed":False,"future_rows_or_tracks":False,"ids_as_model_input":False,"gt_fields_in_feature_tensor":False};atomic_json(outmet/f"b3_joint_{args.tag}_f{fold}.json",obj);atomic_json(done,{"status":"DONE","fold":fold,"tag":args.tag,"checkpoint":str(cp.resolve()),"metrics":str((outmet/f"b3_joint_{args.tag}_f{fold}.json").resolve())});best=cp;models[fold]=JointMatcher.load(best)[:4];folds_metrics[str(fold)]=obj
    for fold in range(4):
        cp=outck/f"b3_joint_{args.tag}_f{fold}_step{args.steps:06d}.npz"
        if cp.exists() and fold not in models:models[fold]=JointMatcher.load(cp)[:4]
    if models:
        all_groups:dict[tuple[int,int],list[int]]=defaultdict(list)
        for i,r in enumerate(rows):all_groups[(int(r["video_id"]),int(r["image_id"]))].append(i)
        for k in all_groups:all_groups[k].sort(key=lambda i:(int(float(rows[i].get("proposal_local_id",0) or 0)),int(float(rows[i].get("track_id",0) or 0)),i))
        atomic_json(outmet/f"b3_joint_replay_{args.tag}.json",replay(models,rows,x,all_groups))
    atomic_json(outmet/f"b3_joint_aggregate_{args.tag}.json",{"phase":"Phase83","route":"B3_JOINT_SUPPORT","tag":args.tag,"steps":args.steps,"folds":folds_metrics,"replay":str((outmet/f"b3_joint_replay_{args.tag}.json").resolve()) if (outmet/f"b3_joint_replay_{args.tag}.json").exists() else None,"public_dev_q1_sealed_accessed":False,"controller_run":False});print(json.dumps({"status":"COMPLETE","route":"B3_JOINT_SUPPORT","tag":args.tag,"folds":sorted(models)},indent=2))


if __name__=="__main__":main()
