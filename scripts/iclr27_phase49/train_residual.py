#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, tempfile
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase49.residual import RawPreservingResidualBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase49'
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name); os.close(fd); torch.save(obj,t); os.replace(t,p)
def vec(k,m,f,p=16):
 z=f[np.asarray(m[k]['rows'][:min(p,16)])].mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--tag',default='phase49_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args()
 torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError(f'GPU mismatch expected {a.expected_physical_gpu}, vis={vis}')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=749000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr)
 man=json.loads((ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{a.fold}.json').read_text()); fit=[r for r in man['records'] if r.get('split')=='fit' and r.get('kind')=='multi_positive_cross_video']; steps=100 if a.smoke else a.steps
 run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; comp=OUT/'completion'; ck=OUT/'checkpoints'; atomic(comp/f'{run}.launched',{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu})
 model=RawPreservingResidualBridge().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4); rng=np.random.default_rng(seed+13); hist=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=np.asarray(vec(r['query_track_key'],meta,feats)); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key'); pos=np.asarray([vec(k,meta,feats) for k in sk]) if sk else np.asarray([q]); neg=vec(hk,meta,feats) if hk in meta else q
  qt=torch.tensor(q,device=dev).view(1,-1); stt=torch.tensor(pos,device=dev).unsqueeze(0); z,alpha,res=model(qt,stt,torch.ones(1,len(pos),device=dev,dtype=torch.bool),True); pvec=torch.tensor(pos,device=dev); nvec=torch.tensor(neg,device=dev).view(1,-1); pos_score=(z@pvec.T).max(1).values; neg_score=(z@nvec.T).squeeze(1); raw_pos=(qt@pvec.T).max(1).values; raw_neg=(qt@nvec.T).squeeze(1)
  # Top-1 hard-negative ranking, prefix consistency, and strong raw anchor.
  rank=F.relu(0.10-pos_score+neg_score).mean(); raw_rank=F.relu(0.10-raw_pos.detach()+raw_neg.detach()).mean(); preserve=(z-qt).pow(2).mean(); consistency=F.mse_loss(z,qt.detach()); loss=rank+0.25*raw_rank+0.10*preserve+0.05*consistency+0.001*res.pow(2).mean()
  opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); opt.step()
  if st==1 or st%100==0 or st==steps: hist.append({'step':st,'loss':float(loss.detach().cpu()),'alpha':float(alpha.detach().mean().cpu()),'residual_abs':float(res.detach().abs().mean().cpu())})
  if st%500==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase49_raw_preserving_support_residual','alpha_max':model.alpha_max}; save(ck/f'{run}_latest.pt',payload); save(ck/f'{run}_best.pt',payload)
 atomic(OUT/'metrics'/f'{run}.json',{'phase':49,'fold':a.fold,'steps':steps,'history':hist,'checkpoint_best':str(ck/f'{run}_best.pt'),'protocol':'phase49_raw_preserving_support_residual','amp':'bf16' if dev.type=='cuda' else 'fp32','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(comp/f'{run}.done',{'fold':a.fold,'steps':steps,'checkpoint':str(ck/f'{run}_best.pt')}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(comp/f'{run}.done')}))
if __name__=='__main__': main()
