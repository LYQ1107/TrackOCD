#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata,retrieval
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase34'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); val=[r for r in man['records'] if r['split']=='val']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); supp={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val if r['kind']=='multi_positive_cross_video'}; per={}
  for p in PREFIXES:
   raw=[]; bridge=[]
   for k in keys:
    inds=meta[k]['rows'][:min(p,16)]; q=feats[np.asarray(inds)].mean(0); q/=max(np.linalg.norm(q),1e-8); raw.append(q); ss=[]
    for s in supp.get(k,[]):
     si=meta[s]['rows'][:min(p,16)]; v=feats[np.asarray(si)].mean(0); v/=max(np.linalg.norm(v),1e-8); ss.append(v)
    if ss:
     ss=np.asarray(ss); w=np.exp((ss@q)/0.1); w/=w.sum(); b=.75*q+.25*(w[:,None]*ss).sum(0); b/=max(np.linalg.norm(b),1e-8); bridge.append(b)
    else: bridge.append(q)
   per[str(p)]={'raw':retrieval(keys,np.asarray(raw),meta),'prototype_bridge':retrieval(keys,np.asarray(bridge),meta)}
  folds.append({'fold':fold,'tracklets':len(keys),'prefix':per})
 agg={str(p):{m:float(np.mean([f['prefix'][str(p)]['prototype_bridge'][m] for f in folds])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')} for p in PREFIXES}; result={'protocol':'trackocd_phase34_stage1_reranker_weighted_prototype_bridge','folds':folds,'aggregate':agg,'fixed_parameters':{'top_k':5,'temperature':0.1,'beta':0.25},'mot_invariants':{'continuity':1.0,'duplicate_tracks':0,'fragmentation_delta':0,'parent_mismatch':'0/26946'},'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/GT input']}; atomic(OUT/'metrics/stage1_diagnostic.json',result); atomic(OUT/'completion/stage1.done',{'stage':1,'folds':4}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
