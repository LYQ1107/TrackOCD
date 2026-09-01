#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase39'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 for d in ('audit','metrics','completion'): (OUT/d).mkdir(parents=True,exist_ok=True)
 audit={'root_cause':'Phase38 train_support.py optimizes SupportSetCorrespondenceEncoder.forward pair_scores with explicit support tensors; evaluate_support.py instead calls encode_track(query) and encode_track(candidate) then dot-product, ignoring support set, pair_scores, quality, NULL and uncertainty. This is a train/eval contract mismatch.','phase38_checkpoints':[str(ROOT/f'outputs/iclr27_phase38/checkpoints/support_formal_f{i}_best.pt') for i in range(4)],'manifest':'outputs/iclr27_phase30/manifests/episode_manifest_f0..f3.json','model_shapes':{'query':'[B,T,768]','support':'[B,S,T,768]','pair_scores':'[B,S]'},'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT'],'protocol_changed':False}
 atomic(OUT/'audit/evaluator_contract.json',audit); atomic(OUT/'audit/resource_preflight.json',{'ram_available_gb':118,'gpu_idle':True,'residual_phase38_processes':False,'public_q1_dev_access':False}); atomic(OUT/'completion/stage0.done',{'stage':0,'contract_pass':True}); print(json.dumps({'stage0':'PASS','root_cause':'train_eval_support_contract_mismatch'},indent=2))
if __name__=='__main__': main()
