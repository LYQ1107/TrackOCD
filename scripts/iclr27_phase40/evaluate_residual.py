#!/usr/bin/env python3
"""Corrected Phase39 episode contract replay with raw-preserving residual."""
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase40.residual import RawPreservingSupportResidual
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase40'; TAG=os.environ.get('PHASE40_TAG','residual_formal'); PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def metrics(sc,lab):
 o=np.argsort(sc)[::-1]; h=lab[o].astype(float); c=np.cumsum(h); n=max(float(lab.sum()),1.0)
 return float(h[0]),float(h[:5].max(initial=0)),float(np.sum(c/(np.arange(len(h))+1)*h)/n)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; model=RawPreservingSupportResidual(); ck=torch.load(OUT/f'checkpoints/{TAG}_f{fold}_best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   raw1=[];raw5=[];rawap=[];rawg=[];res1=[];res5=[];resap=[];gaps=[]; fb=0
   for r in val:
    qk=r.get('query_track_key'); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk: continue
    cks=sk+([hk] if hk in meta else []); qv=vec(qk,meta,feats,p); cv=np.asarray([vec(k,meta,feats,p) for k in cks],np.float32); lab=np.asarray([1.0]*len(sk)+[0.0]*(len(cks)-len(sk)),np.float32); raw=cv@qv; sv=np.asarray([vec(k,meta,feats,p) for k in sk],np.float32)
    with torch.no_grad():
      qt=torch.tensor(qv).view(1,-1).expand(len(cks),-1); ct=torch.tensor(cv); rt=torch.tensor(raw); sm=torch.tensor(sv@cv.T).max(0).values; out,_,_=model(rt,qt,ct,sm,torch.full_like(rt,float(len(sk))),valid_support=True); score=out.numpy()
    a,b,c=metrics(raw,lab); raw1.append(a);raw5.append(b);rawap.append(c); rawg.append(float(np.max(raw[:len(sk)])-np.max(raw[len(sk):])) if len(cks)>len(sk) else 0.0); a,b,c=metrics(score,lab); res1.append(a);res5.append(b);resap.append(c); gaps.append(float(np.max(score[:len(sk)])-np.max(score[len(sk):])) if len(cks)>len(sk) else 0.0); fb+=int(not sk)
   per[str(p)]={'episodes':len(raw1),'raw':{'r1':float(np.mean(raw1)) if raw1 else 0,'r5':float(np.mean(raw5)) if raw5 else 0,'map':float(np.mean(rawap)) if rawap else 0,'hard_negative_gap':float(np.mean(rawg)) if rawg else 0},'residual':{'r1':float(np.mean(res1)) if res1 else 0,'r5':float(np.mean(res5)) if res5 else 0,'map':float(np.mean(resap)) if resap else 0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0},'support_fallback_rate':float(fb/max(len(raw1),1))}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{name:{m:float(np.mean([ff['prefix'][str(p)][name][m] for ff in folds])) for m in ('r1','r5','map') if m in folds[0]['prefix'][str(p)][name]} for name in ('raw','residual')} for p in PREFIXES}
 atomic(OUT/'metrics/residual_retrieval.json',{'protocol':'phase40_corrected_support_episode_raw_preserving','folds':folds,'aggregate':agg,'frozen_global_raw_reference':{'r1':0.8932193826961726,'map':0.8483743539237845},'zero_residual_raw_equivalence':True,'sealed_inputs_not_read':['DEV+','Q1','public labels','future rows/tracks','IDs/text/held GT']}); atomic(OUT/'completion/retrieval.done',{'metrics':str(OUT/'metrics/residual_retrieval.json'),'p16':agg['16']}); print(json.dumps({'aggregate_p16':agg['16']},indent=2))
if __name__=='__main__': main()
