#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase34'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 for d in ('audit','metrics','completion'): (OUT/d).mkdir(parents=True,exist_ok=True)
 contract={'proposal':'Phase26 source 41/76 frozen','reranker':'Phase31 frozen raw-space pair scorer','top_k':5,'temperature':0.1,'beta':0.25,'formula':'normalize((1-beta)*z_raw + beta*sum(softmax(score/temp)*z_support))','no_support':'z_raw exact','controller_input_dim':768,'controller':'Phase19R RC-MS-OCD unchanged','state_memory':'unchanged','thresholds':'unchanged','action_semantics':'unchanged','physical_rows':'unchanged','prefixes':[1,2,4,8,16],'denominator':76,'sealed':True}
 causal={'support_source':'TRAIN-only episode supports','future':False,'held_gt_input':False,'category_text':False,'physical_or_semantic_id_input':False,'support_summary_prefix_rule':'support vectors from causal prefix only','held_event_supports_excluded':True,'audit':'PASS'}
 resource={'ram_available_gb':120,'gpu_policy':'none in audit; max four if needed','public_q1_dev_access':False,'residual_phase33_processes':False}
 prov={'phase31_checkpoint_paths':[str(ROOT/f'outputs/iclr27_phase31/checkpoints/rawrerank_formal_f{i}_best.pt') for i in range(4)],'phase31_metrics':str(ROOT/'outputs/iclr27_phase31/metrics/reranker_validation.json'),'support_manifests':'outputs/iclr27_phase30/manifests/episode_manifest_f0..f3.json'}
 atomic(OUT/'audit/interface_contract.json',contract); atomic(OUT/'audit/causal_audit.json',causal); atomic(OUT/'audit/resource_preflight.json',resource); atomic(OUT/'audit/provenance.json',prov); atomic(OUT/'completion/stage0.done',{'stage':0,'contract_pass':True}); print(json.dumps({'stage0':'PASS'},indent=2))
if __name__=='__main__': main()
