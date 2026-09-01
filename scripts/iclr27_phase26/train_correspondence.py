#!/usr/bin/env python3
"""Train one legal TRAIN fold of the sole Phase26 correspondence route."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.iclr27_phase26.protocol import CSV_PATH, FEAT_PATH, P22_MANIFEST, by_track, load_aligned_features, order_key
from src.iclr27_phase26.correspondence import TrackCorrespondenceEncoder, metadata

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/iclr27_phase26"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f: json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent)); os.close(fd)
    try: torch.save(value, tmp); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256(path):
    h=hashlib.sha256();
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def pad_sequences(keys, track_rows, feats, max_len=16):
    arr=np.zeros((len(keys),max_len,feats.shape[1]),np.float32); mask=np.zeros((len(keys),max_len),bool)
    for i,k in enumerate(keys):
        inds=track_rows[k][-max_len:]; x=feats[inds]; arr[i,:len(x)]=x; mask[i,:len(x)]=True
    return arr,mask


def retrieval(model, keys, track_rows, feats, cat, video, device):
    if not keys: return {"queries":0,"r1":0.,"r5":0.,"map":0.,"pairs":0}
    vals=[]
    for st in range(0,len(keys),64):
        x,m=pad_sequences(keys[st:st+64],track_rows,feats); vals.append(model(torch.from_numpy(x).to(device),torch.from_numpy(m).to(device)).cpu().numpy())
    emb=np.concatenate(vals,0); r1=[]; r5=[]; ap=[]; pairs=0
    for i,k in enumerate(keys):
        cand=[j for j,q in enumerate(keys) if j!=i and int(video[q])!=int(video[k])]; pos=[j for j in cand if int(cat[q:=keys[j]])==int(cat[k])]
        if not pos: continue
        order=np.asarray(cand)[np.argsort(emb[cand]@emb[i])[::-1]]; hits=np.asarray([int(j in pos) for j in order]); pairs+=len(cand); r1.append(float(hits[:1].max(initial=0))); r5.append(float(hits[:5].max(initial=0))); cum=np.cumsum(hits); ap.append(float(np.sum(cum/(np.arange(len(hits))+1)*hits)/max(len(pos),1)))
    return {"queries":len(r1),"r1":float(np.mean(r1)) if r1 else 0.,"r5":float(np.mean(r5)) if r5 else 0.,"map":float(np.mean(ap)) if ap else 0.,"pairs":pairs}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--fold",type=int,required=True); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--expected-physical-gpu",type=int,default=-1); ap.add_argument("--steps",type=int,default=2000); ap.add_argument("--batch-size",type=int,default=16); ap.add_argument("--checkpoint-every",type=int,default=500); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--tag",default="correspondence")
    a=ap.parse_args(); torch.set_num_threads(2); vis=os.environ.get("CUDA_VISIBLE_DEVICES","")
    if a.expected_physical_gpu>=0 and vis and vis.split(",")[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError(f"expected physical GPU {a.expected_physical_gpu}, CUDA_VISIBLE_DEVICES={vis}")
    device=torch.device(a.device if torch.cuda.is_available() else "cpu");
    if device.type=="cuda": torch.cuda.set_device(device)
    seed=20261001+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows=list(csv.DictReader(CSV_PATH.open(newline="",encoding="utf-8"))); cls,roi,alignment=load_aligned_features(rows); feats=(.8*cls+.2*roi).astype(np.float32); feats/=np.maximum(np.linalg.norm(feats,axis=1,keepdims=True),1e-6)
    tracks=by_track(rows); cat={k:int(rows[v[-1]].get("gt_category_id_common",-1)) for k,v in tracks.items()}; video={k:int(rows[v[-1]]["video_id"]) for k,v in tracks.items()}; manifest=json.loads(P22_MANIFEST.read_text()); fr=next(x for x in manifest["folds"] if int(x["fold"])==a.fold); fit_v=set(map(int,fr["fit_videos"])); val_v=set(map(int,fr["validation_videos"])); fit_c=set(map(int,fr["fit_categories"])); held_c=set(map(int,fr["held_categories"]))
    fit_by=defaultdict(list); val_by=defaultdict(list)
    for k in tracks:
        if cat[k] in fit_c and video[k] in fit_v and cat[k]>=0: fit_by[cat[k]].append(k)
        if cat[k] in held_c and video[k] in val_v and cat[k]>=0: val_by[cat[k]].append(k)
    fit_by={c:v for c,v in fit_by.items() if len({video[k] for k in v})>=2}; val_keys=sorted([k for v in val_by.values() for k in v]); fit_cats=sorted(fit_by); all_cats=sorted(set(fit_cats))
    run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/"completion"/f"{run}.launched"; done=OUT/"completion"/f"{run}.done"; ckdir=OUT/"checkpoints"; latest=ckdir/f"{run}_latest.pt"; bestp=ckdir/f"{run}_best.pt"; logp=OUT/"logs"/f"{run}.jsonl"
    if done.exists() and not a.resume: print(json.dumps({"status":"already_done","done":str(done)})); return
    if marker.exists() and not a.resume: raise RuntimeError(f"refusing relaunch with marker {marker}")
    marker.write_text(json.dumps({"fold":a.fold,"pid":os.getpid(),"started":time.time(),"device":str(device),"physical_gpu":a.expected_physical_gpu})+"\n")
    model=TrackCorrespondenceEncoder(); model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); start=0; best=-1.; best_step=0; history=[]; rng=np.random.default_rng(seed+17); steps=2 if a.smoke else a.steps; amp=torch.bfloat16 if device.type=="cuda" else None
    if a.resume and latest.exists():
        ck=torch.load(latest,map_location="cpu",weights_only=False); model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start=int(ck.get("global_step",0)); best=float(ck.get("best_score",-1.)); best_step=int(ck.get("best_step",0))
    def sample_key(c):
        ks=fit_by[c]; return ks[int(rng.integers(len(ks)))]
    t0=time.time(); logp.parent.mkdir(parents=True,exist_ok=True)
    for step in range(start+1,steps+1):
        anchors=[]; positives=[]; negatives=[]
        for _ in range(a.batch_size):
            c=int(rng.choice(all_cats)); ks=fit_by[c]; x=sample_key(c); xv=video[x]; pks=[k for k in ks if video[k]!=xv] or ks; p=pks[int(rng.integers(len(pks)))]; nc=int(rng.choice([z for z in all_cats if z!=c])); n=sample_key(nc); anchors.append(x); positives.append(p); negatives.append(n)
        xa,ma=pad_sequences(anchors,tracks,feats); xp,mp=pad_sequences(positives,tracks,feats); xn,mn=pad_sequences(negatives,tracks,feats); va=torch.from_numpy(xa).to(device); vp=torch.from_numpy(xp).to(device); vn=torch.from_numpy(xn).to(device); ma=torch.from_numpy(ma).to(device); mp=torch.from_numpy(mp).to(device); mn=torch.from_numpy(mn).to(device); opt.zero_grad(set_to_none=True)
        ctx=torch.autocast(device_type="cuda",dtype=amp) if amp is not None else torch.autocast(device_type="cpu",enabled=False)
        with ctx:
            ea=model(va,ma); ep=model(vp,mp); en=model(vn,mn); pos_sim=(ea*ep).sum(-1); neg_sim=(ea*en).sum(-1); loss_rank=F.relu(.20-pos_sim+neg_sim).mean(); loss_pos=(1-pos_sim).mean(); loss=loss_rank+.10*loss_pos
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite correspondence loss step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); rec={"step":step,"loss":float(loss.detach().cpu()),"rank_loss":float(loss_rank.detach().cpu()),"pos_similarity":float(pos_sim.detach().mean().cpu()),"neg_similarity":float(neg_sim.detach().mean().cpu())}
        if step%a.checkpoint_every==0 or step==steps:
            valm=retrieval(model,val_keys,tracks,feats,cat,video,device); score=float(valm["r1"]+.2*valm["r5"]+.1*valm["map"]); payload={"model":{k:v.detach().cpu() for k,v in model.state_dict().items()},"optimizer":opt.state_dict(),"global_step":step,"best_score":best,"best_step":best_step,"fold":a.fold,"seed":seed,"metadata":metadata(model),"protocol":"trackocd_iclr27_phase26_correspondence_encoder","feature_alignment":alignment,"source_csv_sha256":sha256(CSV_PATH),"feature_sha256":sha256(FEAT_PATH),"fit_categories":all_cats,"validation_categories":sorted(val_by),"validation_videos":sorted(val_v),"amp":"bf16" if amp is not None else "fp32"}; atomic_torch(latest,payload); atomic_torch(ckdir/f"{run}_step{step:05d}.pt",payload)
            if score>best: best,best_step=score,step; payload["best_score"],payload["best_step"]=best,best_step; atomic_torch(bestp,payload)
            rec.update({"validation":valm,"validation_score":score,"best_score":best,"elapsed_s":time.time()-t0}); history.append(rec); logp.open("a",encoding="utf-8").write(json.dumps(rec,sort_keys=True)+"\n")
    final=retrieval(model,val_keys,tracks,feats,cat,video,device); result={"protocol":"trackocd_iclr27_phase26_correspondence_training","fold":a.fold,"tag":a.tag,"seed":seed,"steps":steps,"smoke":bool(a.smoke),"device":str(device),"physical_gpu":a.expected_physical_gpu,"amp":"bf16" if amp is not None else "fp32","fit_tracklets":sum(map(len,fit_by.values())),"fit_categories":all_cats,"validation_tracklets":len(val_keys),"validation_metrics":final,"best_score":best,"best_step":best_step,"history":history,"checkpoint_best":str(bestp),"checkpoint_latest":str(latest),"marker":str(marker),"done":str(done),"metadata":metadata(model),"sealed_inputs_not_read":["DEV+","Q1","public new-model labels","future frames/tracks","physical/semantic IDs","semantic text"]}; atomic_json(OUT/"metrics"/f"{run}.json",result); done.write_text(json.dumps({"fold":a.fold,"steps":steps,"checkpoint":str(bestp),"validation":final},sort_keys=True)+"\n"); print(json.dumps({"fold":a.fold,"steps":steps,"val_r1":final["r1"],"val_r5":final["r5"],"best_step":best_step,"done":str(done)},indent=2,sort_keys=True))


if __name__=="__main__": main()
