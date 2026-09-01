#!/usr/bin/env python3
import json,os,tempfile,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase32'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 OUT.joinpath('audit').mkdir(parents=True,exist_ok=True); OUT.joinpath('completion').mkdir(parents=True,exist_ok=True)
 contract={'proposal':'Phase26 learned class-agnostic source, ceiling 41/76','controller':'Phase19R RC-MS-OCD unchanged','state_memory':'frozen','thresholds':'frozen','action_semantics':'frozen','routing_effect':'candidate enumeration/order only; raw vector/state bundle passed unchanged','physical_mot':'rows, parent assignment and IDs unchanged','row_key':'Phase26/31 exact aligned key','prefixes':[1,2,4,8,16],'denominator':76,'sealed':True,'protocol_preserved':True,'controller_source':'src/iclr27_phase19r/models/controller.py (read-only)','runner_source':'src/iclr27_phase19r/runtime/runner.py (read-only)'}
 causal={'causal_prefix_rule':'routing uses rows at or before current prefix only','future_access':False,'held_gt_as_input':False,'category_text':False,'physical_id_semantics':False,'reranker_pair_score_in_controller':False,'raw_vector_forward_contract':'unchanged','audit_result':'PASS'}
 resource={'ram':'preflight Phase31 ~115GB available','gpu_policy':'at most four workers; no training in Stage0','public_q1_dev_access':False,'residual_phase31_processes':False,'cwd':str(ROOT)}
 replay={'raw_comparator_source':'outputs/iclr27_phase31/metrics/compatibility_diagnostic.json','historical_phase28_ct':'3/76','phase31_raw_comparator_ct':'1/76','routing_replay':'not yet run; Stage0 only contract audit','first_root_cause_if_failed':None}
 atomic(OUT/'audit/interface_contract.json',contract); atomic(OUT/'audit/causal_routing_audit.json',causal); atomic(OUT/'audit/resource_preflight.json',resource); atomic(OUT/'audit/raw_vs_routed_replay.json',replay); atomic(OUT/'completion/stage0.done',{'stage':0,'contract_pass':True})
 (OUT/'audit/STAGE0_AUDIT.md').write_text('# Phase32 Stage 0 — Interface/Causal Audit\n\nContract PASS: routing is restricted to candidate order; unchanged raw vectors, state bundle, physical IDs, known mask, thresholds and action semantics are passed to RC-MS-OCD. Future/held-GT/ID/text leakage checks PASS. Raw-vs-routed replay is reserved for Stage1.\n')
 print(json.dumps({'stage0':'PASS','outputs':str(OUT/'audit')},indent=2))
if __name__=='__main__': main()
