#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata,retrieval
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase36'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json'%fold))); keys=sorted({r['query_track_key'] for r in man['records'] if r['split']=='val' and r['query_track_key'] in meta}); per={}
  for p in PREFIXES:
   raw=[]; gated=[]; low=0
   for k in keys:
    inds=meta[k]['rows'][:min(p,16)]; x=feats[np.asarray(inds)]; cur=x[-1]; cur/=max(np.linalg.norm(cur),1e-8); raw.append(cur); 
    if len(x)<2: gated.append(cur); low+=1; continue
    h=x[-4:].mean(0); h/=max(np.linalg.norm(h),1e-8); stab=float(np.mean(np.sum(x[:-1]*x[1:],axis=1))); alpha=float(0.25/(1+np.exp(-4*(stab-0.7)))); z=(1-alpha)*cur+alpha*h; z/=max(np.linalg.norm(z),1e-8); gated.append(z)
   per[str(p)]={'raw':retrieval(keys,np.asarray(raw),meta),'reliability_gate':retrieval(keys,np.asarray(gated),meta),'low_reliability_count':low}
  folds.append({'fold':fold,'tracklets':len(keys),'prefix':per})
 vals=[f['prefix']['16']['reliability_gate'] for f in folds]; agg={m:float(np.mean([v[m] for v in vals])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')}; atomic(OUT/'metrics/stage1_reliability.json',{'protocol':'trackocd_phase36_reliability_gated_history_diagnostic','folds':folds,'aggregate_p16':agg,'fixed_gate':'alpha=0.25*sigmoid(4*(stability-0.7)); K=4','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/GT']}); atomic(OUT/'completion/stage1.done',{'stage':1,'folds':4}); print(json.dumps({'p16':agg},indent=2))
if __name__=='__main__': main()
