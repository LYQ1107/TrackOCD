#!/usr/bin/env python3
import argparse,json,os,tempfile,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase38'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats,p):
 x=feats[np.asarray(meta[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(np.linalg.norm(z),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--batch-size',type=int,default=64); ap.add_argument('--checkpoint-every',type=int,default=500); ap.add_argument('--tag',default='support_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args(); torch.set_num_threads(1)
 vis=os.environ.get('CUDA_VISIBLE_DEVICES','');
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=383000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); man=json.load(open(ROOT/('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json'%a.fold))); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video'];
 run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json';
 for d in (marker.parent,latest.parent,metrics.parent): d.mkdir(parents=True,exist_ok=True)
 atomic(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu}); model=SupportSetCorrespondenceEncoder().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4); rng=np.random.default_rng(seed+7); steps=100 if a.smoke else a.steps; hist=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=vec(r['query_track_key'],meta,feats,16); ss=[vec(k,meta,feats,16) for k in r.get('support_track_keys',[]) if k in meta] or [q]; h=vec(r['hard_negative_track_key'],meta,feats,16) if r.get('hard_negative_track_key') in meta else q; qt=torch.tensor(q,device=dev).view(1,1,-1); stt=torch.tensor(np.asarray(ss),device=dev).view(1,len(ss),1,-1); hm=torch.ones((1,len(ss),1),device=dev,dtype=torch.bool); qm=torch.ones((1,1),device=dev,dtype=torch.bool); sm=torch.ones((1,len(ss),1),device=dev,dtype=torch.bool); setm=torch.ones((1,len(ss)),device=dev,dtype=torch.bool); out=model(qt,qm,stt,sm,setm); he=torch.sum(out['query_embedding']*model.encode_track(torch.tensor(h,device=dev).view(1,1,-1),qm),dim=-1); pos=out['pair_scores'].max(dim=1).values; loss=F.relu(0.15-pos+he).mean()+0.2*F.binary_cross_entropy_with_logits(out['null_logit'],torch.zeros_like(out['null_logit']))+0.01*sum((p.float()**2).mean() for p in model.parameters()); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
  if st%max(1,min(a.checkpoint_every,steps))==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase38_prior_completed_support_correspondence'}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu())})
 atomic(metrics,{'fold':a.fold,'steps':steps,'history':hist,'checkpoint_best':str(best),'protocol':'phase38_prior_completed_support_correspondence'}); atomic(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(done)},indent=2))
if __name__=='__main__': main()
