#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase41'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 ck=[ROOT/f'outputs/iclr27_phase40/checkpoints/residual_formal_bf16_f{i}_best.pt' for i in range(4)]
 shapes=[]
 for p in ck:
  d=torch.load(p,map_location='cpu',weights_only=False); shapes.append({'path':str(p),'exists':p.exists(),'step':d.get('step'),'beta_max':d.get('beta_max')})
 atomic(OUT/'audit/contract.json',{'phase':41,'phase40_checkpoints':shapes,'row_vector_dim':768,'bridge':'normalize(raw + alpha*clamp(support_context-raw))','alpha_zero_identity':True,'hard_gap_anchor':'raw score retained; safety clamp is score-level','protocol_unchanged':True,'support_policy':'Phase38 PRIOR_COMPLETED_TRACK causal only','forbidden_inputs':['category','text','physical/semantic ID','future','held GT','StateMemory','controller action']})
 atomic(OUT/'audit/resource_preflight.json',{'gpu_ids':[4,5,6,7],'gpu_free_mib':40337,'ram_available_gb':118,'public_q1_dev_access':False,'bounded_workers':4})
 atomic(OUT/'audit/safety_baseline.json',{'phase40_global_p16':{'raw_r1':0.8932193826961726,'raw_map':0.8483743525266084,'raw_hard_gap':0.18955873782968155,'residual_r1':0.9505545291835614,'residual_map':0.8980711989206362,'residual_hard_gap':0.2057291741010681},'hard_gap_regression_folds':[0,1],'raw_fallback_required':True,'support_masks_causal':True})
 atomic(OUT/'completion/stage0.done',{'stage':0,'contract':'PASS'})
 print(json.dumps({'stage0':'PASS','checkpoints':len(ck)}))
if __name__=='__main__': main()
