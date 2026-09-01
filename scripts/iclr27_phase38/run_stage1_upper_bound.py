#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase38'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); events=[json.loads(x) for x in open(ROOT/'outputs/iclr27_phase19r/manifests/held_known_positive_events.jsonl') if x.strip()]; recs=[]
 for e in events:
  cat=int(e['category_gt_denominator_only']); tv=int(e['target_video']); prior=[k for k,m in meta.items() if m['video']<tv and m['length']>=2]; same=[k for k in prior if meta[k]['category']==cat];
  recs.append({'event_key':e['event_key'],'fold':e['fold'],'category':cat,'target_video':tv,'prior_support_count':len(prior),'same_category_support_count':len(same),'support_present':bool(same),'oracle_ceiling':int(bool(same))})
 summary={'events':76,'support_present':sum(r['support_present'] for r in recs),'oracle_ceiling':sum(r['oracle_ceiling'] for r in recs),'fold_support':{str(f):sum(r['oracle_ceiling'] for r in recs if r['fold']==f) for f in range(4)},'category_coverage':len(set(r['category'] for r in recs if r['oracle_ceiling'])),'video_coverage':len(set(r['target_video'] for r in recs if r['oracle_ceiling'])),'policy':'PRIOR_COMPLETED_TRACK','diagnostic_only':True}
 atomic(OUT/'metrics/support_upper_bound.json',{'summary':summary,'records':recs,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text model input']}); atomic(OUT/'completion/stage1.done',{'stage':1,'upper_bound':summary}); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
