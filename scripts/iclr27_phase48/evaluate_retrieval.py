#!/usr/bin/env python3
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase48.correspondence import SupportConditionedEncoder
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase48'; PREFIXES=(1,2,4,8,16)
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats,p):
 z=feats[np.asarray(meta[k]['rows'][:min(p,16)])].mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
@torch.no_grad()
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); folds=[]
 for fold in range(4):
  man=json.loads((ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json').read_text()); val=[r for r in man['records'] if r.get('split')=='val' and r.get('kind')=='multi_positive_cross_video']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); support={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val if r['query_track_key'] in meta}; ck=torch.load(OUT/f'checkpoints/phase48_formal_f{fold}_best.pt',map_location='cpu',weights_only=False); model=SupportConditionedEncoder().to(dev); model.load_state_dict(ck['model']); model.eval(); vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys]); prefix_out={}
  for p in PREFIXES:
   base=np.asarray([vec(k,meta,feats,p) for k in keys],np.float32); raw_sim=base@base.T; learned=[]; raw=[]; gaps=[]
   for i,k in enumerate(keys):
    cand=np.where((np.arange(len(keys))!=i)&(vids!=vids[i]))[0]; pos=cand[cats[cand]==cats[i]]; neg=cand[cats[cand]!=cats[i]]
    if len(pos)==0 or len(neg)==0: continue
    s=[vec(x,meta,feats,p) for x in support.get(k,[]) if x in meta]; s=s or [base[i]]; q=torch.tensor(base[i],device=dev).view(1,-1); st=torch.tensor(np.asarray(s),device=dev).unsqueeze(0); out=model(q,st,torch.ones((1,len(s)),device=dev,dtype=torch.bool)); cv=model.encode(torch.tensor(base[cand],device=dev)); scores=(out['embedding']@cv.T).cpu().numpy()[0]; rs=raw_sim[i,cand]; order=np.argsort(scores)[::-1]; ro=np.argsort(rs)[::-1]; hit=np.isin(cand[order],pos).astype(float); rh=np.isin(cand[ro],pos).astype(float); learned.append((float(hit[0]),float(hit[:5].max(initial=0)),float(np.sum(np.cumsum(hit)/(np.arange(len(hit))+1)*hit)/max(len(pos),1)))); raw.append((float(rh[0]),float(rh[:5].max(initial=0)),float(np.sum(np.cumsum(rh)/(np.arange(len(rh))+1)*rh)/max(len(pos),1)))); gaps.append(float(scores[np.isin(cand,pos)].max()-scores[np.isin(cand,neg)].max()))
   prefix_out[str(p)]={'queries':len(learned),'raw':{'r1':float(np.mean([x[0] for x in raw])) if raw else 0,'r5':float(np.mean([x[1] for x in raw])) if raw else 0,'map':float(np.mean([x[2] for x in raw])) if raw else 0},'learned':{'r1':float(np.mean([x[0] for x in learned])) if learned else 0,'r5':float(np.mean([x[1] for x in learned])) if learned else 0,'map':float(np.mean([x[2] for x in learned])) if learned else 0,'hard_gap':float(np.mean(gaps)) if gaps else 0}}
  folds.append({'fold':fold,'queries':len(keys),'prefix':prefix_out})
 agg={};
 for p in PREFIXES:
  fs=[f['prefix'][str(p)] for f in folds]; agg[str(p)]={'raw':{m:float(np.mean([x['raw'][m] for x in fs])) for m in ('r1','r5','map')},'learned':{m:float(np.mean([x['learned'][m] for x in fs])) for m in ('r1','r5','map')},'hard_gap':float(np.mean([x['learned']['hard_gap'] for x in fs]))}
 result={'phase':48,'protocol':'phase48_support_conditioned_retrieval','prefixes':list(PREFIXES),'folds':folds,'aggregate':agg,'gate_r48': 'FAIL','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}; atomic(OUT/'metrics/phase48_retrieval.json',result); atomic(OUT/'completion/retrieval.done',{'phase':48,'gate':'FAIL'}); print(json.dumps(agg['16'],indent=2))
if __name__=='__main__': main()
