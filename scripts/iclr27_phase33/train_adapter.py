#!/usr/bin/env python3
import argparse,json,os,tempfile,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase33.adapter import QueryConditionedAdapter
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase33'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats,p=16):
 x=feats[np.asarray(meta[k]['rows'])[:p]]; return x.mean(0)/max(np.linalg.norm(x.mean(0)),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--checkpoint-every',type=int,default=500); ap.add_argument('--tag',default='adapter_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args(); torch.set_num_threads(1)
 vis=os.environ.get('CUDA_VISIBLE_DEVICES','');
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=333000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % a.fold))); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video'];
 run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json';
 for d in (marker.parent, latest.parent, metrics.parent): d.mkdir(parents=True,exist_ok=True)
 atomic(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu}); model=QueryConditionedAdapter().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4); rng=np.random.default_rng(seed+4); steps=100 if a.smoke else a.steps; hist=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=vec(r['query_track_key'],meta,feats); ss=[vec(k,meta,feats) for k in r.get('support_track_keys',[]) if k in meta]; h=vec(r['hard_negative_track_key'],meta,feats) if r.get('hard_negative_track_key') in meta else q; qt=torch.tensor(q,device=dev).unsqueeze(0); s=torch.tensor(np.asarray(ss or [q]),device=dev).unsqueeze(0); ht=torch.tensor(h,device=dev).unsqueeze(0); qe=model(qt,s); pe=torch.sum(qe*model(s[:,0,:].unsqueeze(1),None).squeeze(1),dim=-1); he=torch.sum(qe*model(ht,None),dim=-1); loss=F.relu(0.15-pe+he).mean()+0.05*(1-torch.sum(qe*qt,dim=-1)).mean()+0.01*model.delta[-1].weight.pow(2).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
  if st%max(1,min(a.checkpoint_every,steps))==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase33_query_conditioned_feature_adapter'}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu()),'alpha':float((0.2*torch.sigmoid(model.alpha)).detach().cpu())})
 atomic(metrics,{'fold':a.fold,'steps':steps,'history':hist,'checkpoint_best':str(best),'protocol':'phase33_query_conditioned_feature_adapter'}); atomic(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(done)},indent=2))
if __name__=='__main__': main()
