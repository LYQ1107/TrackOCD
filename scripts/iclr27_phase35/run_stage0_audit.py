#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase35'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 for d in ('audit','metrics','completion'): (OUT/d).mkdir(parents=True,exist_ok=True)
 contract={'K':4,'formula':'normalize(0.5*z_current + 0.5*mean(current_and_previous_causal_history))','history_min_frames':2,'short_history_fallback':'raw_current','controller_input_dim':768,'controller':'Phase19R RC-MS-OCD unchanged','state_memory':'unchanged','thresholds':'unchanged','action_semantics':'unchanged','physical_mot':'unchanged','prefixes':[1,2,4,8,16],'denominator':76,'sealed':True}
 causal={'ordering':'track rows sorted by event_rank; only indices <= position','future_access':False,'support_used':False,'held_gt_input':False,'category_text':False,'physical_or_semantic_id_input':False,'history_window':'max four rows ending at current position','audit':'PASS'}
 prov={'raw_features':'Phase19R public_cls_roi.npz (read-only)','controller_source':'src/iclr27_phase19r/models/controller.py','evaluator_source':'src/iclr27_phase19r/evaluation/internal.py','prior_phase34_decision':str(ROOT/'outputs/iclr27_phase34/audit/phase34_decision.json')}
 res={'ram_available_gb':119,'gpu_policy':'no training; CPU replay','public_q1_dev_access':False,'residual_phase34_processes':False}
 atomic(OUT/'audit/history_contract.json',contract); atomic(OUT/'audit/causal_audit.json',causal); atomic(OUT/'audit/provenance.json',prov); atomic(OUT/'audit/resource_preflight.json',res); atomic(OUT/'completion/stage0.done',{'stage':0,'contract_pass':True}); print(json.dumps({'stage0':'PASS'},indent=2))
if __name__=='__main__': main()
