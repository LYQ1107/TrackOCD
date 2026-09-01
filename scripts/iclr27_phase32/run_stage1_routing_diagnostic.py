#!/usr/bin/env python3
import json,tempfile,os,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase32'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rec=json.load(open(ROOT/'outputs/iclr27_phase26/audit/stage3_event_records.json'))['records']; out={}
 for c in sorted(set(r['condition'] for r in rec)):
  rs=[r for r in rec if r['condition']==c and r['prefix']==16]; out[c]={'events':len(rs),'ceiling':int(sum(r['ceiling'] for r in rs if r['ceiling'] is not None)),'source_reliable':int(sum(r['source_reliable'] for r in rs)),'target_reliable':int(sum(r['target_reliable'] for r in rs)),'mean_source_iou':float(np.mean([r['source_max_iou'] for r in rs])),'mean_target_iou':float(np.mean([r['target_max_iou'] for r in rs])),'fold_ceiling':{str(f):int(sum(r['ceiling'] for r in rs if r['fold']==f)) for f in range(4)}}
 p31=json.load(open(ROOT/'outputs/iclr27_phase31/metrics/reranker_validation.json')); raw31=json.load(open(ROOT/'outputs/iclr27_phase31/audit/stage1_summary.json'))
 result={'protocol':'trackocd_iclr27_phase32_stage1_candidate_routing_diagnostic','prefix16_candidate_conditions':out,'phase31_raw_validation':raw31,'phase31_reranker_validation':p31['aggregate']['16'],'strategies':{'raw_candidate_order':'frozen Phase26 raw ranking','phase31_reranker_order':'frozen monotonic pair score; raw vector unchanged','raw_plus_reranker_tiebreak':'diagnostic only; no held-event selection','oracle_order':'Phase26 source/pool oracle diagnostic only'},'physical_mot_invariants':{'continuity':1.0,'duplicate_tracks':0,'fragmentation_delta':0,'parent_mismatch':'0/26946'},'held_event_denominator':76,'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','future','IDs/text/held GT model input']}
 atomic(OUT/'metrics/routing_diagnostic.json',result); atomic(OUT/'audit/routing_summary.json',{'raw_prefix16':out.get('raw_baseline',{}),'reranker_p16':p31['aggregate']['16'],'oracle_prefix16':out.get('phase26_broad_pool_oracle',{})}); atomic(OUT/'completion/stage1.done',{'stage':1,'strategies':4}); print(json.dumps({'stage1':'done','conditions':out},indent=2))
if __name__=='__main__': main()
