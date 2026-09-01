#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random, tempfile
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase48.correspondence import SupportConditionedEncoder

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase48'
def atomic_json(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def save_ckpt(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name); os.close(fd); torch.save(obj,t); os.replace(t,p)
def vec(k,meta,feats,p=16):
 inds=meta[k]['rows'][:min(p,16)]; z=feats[np.asarray(inds)].mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--tag',default='phase48_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args()
 torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','')
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError(f'GPU mismatch expected {a.expected_physical_gpu}, visible {vis}')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=648000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); man=json.loads((ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{a.fold}.json').read_text()); fit=[r for r in man['records'] if r.get('split')=='fit' and r.get('kind')=='multi_positive_cross_video']
 run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; comp=OUT/'completion'; ckdir=OUT/'checkpoints'; marker=comp/f'{run}.launched'; done=comp/f'{run}.done'; metric=OUT/'metrics'/f'{run}.json'; latest=ckdir/f'{run}_latest.pt'; best=ckdir/f'{run}_best.pt'; marker.parent.mkdir(parents=True,exist_ok=True); atomic_json(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu,'protocol':'phase48'})
 model=SupportConditionedEncoder().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4); rng=np.random.default_rng(seed+9); steps=100 if a.smoke else a.steps; history=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=torch.tensor(vec(r['query_track_key'],meta,feats),device=dev).view(1,-1); ss=[vec(k,meta,feats) for k in r.get('support_track_keys',[]) if k in meta]; ss=ss or [q.detach().cpu().numpy()[0]]; h=vec(r['hard_negative_track_key'],meta,feats) if r.get('hard_negative_track_key') in meta else ss[0]; s=torch.tensor(np.asarray(ss),device=dev).unsqueeze(0); hm=torch.ones((1,len(ss)),device=dev,dtype=torch.bool); out=model(q,s,hm); neg=model.encode(torch.tensor(h,device=dev).view(1,-1)); pos=out['pair_scores'].max(1).values; hard=(out['query_embedding']*neg).sum(-1); loss=F.relu(0.15-pos+hard).mean()+0.2*F.mse_loss(out['embedding'],out['query_embedding'].detach())+0.01*sum((p.float()**2).mean() for p in model.parameters()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
  if st==1 or st%100==0 or st==steps: history.append({'step':st,'loss':float(loss.detach().cpu())})
  if st%500==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase48_support_supervision_contract'}; save_ckpt(latest,payload); save_ckpt(best,payload)
 atomic_json(metric,{'fold':a.fold,'steps':steps,'history':history,'checkpoint_best':str(best),'protocol':'phase48_support_supervision_contract','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic_json(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(done)}))
if __name__=='__main__': main()
