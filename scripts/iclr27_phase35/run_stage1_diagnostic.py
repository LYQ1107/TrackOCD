#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata,retrieval
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase35'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); keys=sorted({r['query_track_key'] for r in man['records'] if r['split']=='val' and r['query_track_key'] in meta}); per={}
  for p in PREFIXES:
   raw=[]; hist=[]
   for k in keys:
    inds=meta[k]['rows'][:min(p,16)]; x=feats[np.asarray(inds)]; q=x[-1]; raw.append(q); win=x[-4:]; h=win.mean(0); h/=max(np.linalg.norm(h),1e-8); z=(.5*q+.5*h); z/=max(np.linalg.norm(z),1e-8); hist.append(z)
   per[str(p)]={'raw':retrieval(keys,np.asarray(raw),meta),'history_k4':retrieval(keys,np.asarray(hist),meta),'history_min2_fallback_count':sum(len(meta[k]['rows'][:min(p,16)])<2 for k in keys)}
  folds.append({'fold':fold,'tracklets':len(keys),'prefix':per})
 vals=[f['prefix']['16']['history_k4'] for f in folds]; agg={m:float(np.mean([v[m] for v in vals])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')}; atomic(OUT/'metrics/stage1_history.json',{'protocol':'trackocd_phase35_history_bridge_retrieval','folds':folds,'aggregate_p16':agg,'fixed_K':4,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/GT input']}); atomic(OUT/'completion/stage1.done',{'stage':1,'folds':4}); print(json.dumps({'p16':agg},indent=2))
if __name__=='__main__': main()
