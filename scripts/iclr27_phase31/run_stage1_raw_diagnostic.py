#!/usr/bin/env python3
import json, tempfile, os
from pathlib import Path
import numpy as np
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata, retrieval
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase31'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows,tracks,feats=load_tracks(); meta=track_metadata(rows,tracks); folds=[]
 for fold in range(4):
  man=json.load(open(OUT.parent / ('iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); keys=sorted({r['query_track_key'] for r in man['records'] if r['split']=='val' and r['query_track_key'] in meta}); models={}
  for p in PREFIXES:
   vec=[]; stab=[]
   for k in keys:
    inds=meta[k]['rows'][:min(p,16)]; x=feats[np.asarray(inds)]; v=x.mean(0); v/=max(np.linalg.norm(v),1e-8); vec.append(v); stab.append(float(np.mean(np.sum(x[:-1]*x[1:],axis=1))) if len(x)>1 else 1.0)
   vec=np.asarray(vec,np.float32); stab=np.asarray(stab,np.float32)
   base=retrieval(keys,vec,meta)
   # Fixed, preregistered causal stability adjustment; score remains dominated by raw cosine.
   sim=vec@vec.T; vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys]); allidx=np.arange(len(keys)); r1=[]; r5=[]; aps=[]; gaps=[]
   for i in range(len(keys)):
    cand=allidx[(allidx!=i)&(vids!=vids[i])]; pos=cand[cats[cand]==cats[i]]; neg=cand[cats[cand]!=cats[i]]
    if len(pos)==0 or len(neg)==0: continue
    score=sim[i,cand]+0.05*(stab[i]+stab[cand])/2; order=cand[np.argsort(score)[::-1]]; hit=np.array([j in set(pos.tolist()) for j in order],float); r1.append(float(hit[:1].max(initial=0))); r5.append(float(hit[:5].max(initial=0))); cum=np.cumsum(hit); aps.append(float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(len(pos),1))); gaps.append(float(np.max(score[np.isin(cand,pos)])-np.max(score[np.isin(cand,neg)])))
   models[str(p)]={'raw_cosine':base,'raw_plus_fixed_temporal_stability':{'queries':len(r1),'r1':float(np.mean(r1)) if r1 else 0.0,'r5':float(np.mean(r5)) if r5 else 0.0,'map':float(np.mean(aps)) if aps else 0.0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0.0,'positive_coverage':float(len(r1)/max(len(keys),1))}}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':models})
 result={'protocol':'trackocd_iclr27_phase31_stage1_raw_space_diagnostic','folds':folds,'raw_baseline_reference':'outputs/iclr27_phase30/metrics/stage1_diagnostics.json','no_training':True,'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','held event outcomes','future rows/tracks','IDs/text/GT model input']}
 atomic(OUT/'metrics/stage1_raw_diagnostic.json',result); atomic(OUT/'completion/stage1.done',{'stage':1,'folds':4})
 # aggregate p16
 vals=[f['prefix']['16']['raw_cosine'] for f in folds]; vals2=[f['prefix']['16']['raw_plus_fixed_temporal_stability'] for f in folds]
 summary={'raw_cosine_p16':{k:float(np.mean([v[k] for v in vals])) for k in ('r1','r5','map','hard_negative_gap','positive_coverage')},'raw_plus_fixed_temporal_stability_p16':{k:float(np.mean([v[k] for v in vals2])) for k in ('r1','r5','map','hard_negative_gap','positive_coverage')}}
 atomic(OUT/'audit/stage1_summary.json',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
