#!/usr/bin/env python3
import argparse,json,os,tempfile,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase41'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p=16):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=2000); ap.add_argument('--checkpoint-every',type=int,default=500); ap.add_argument('--tag',default='gate_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args(); torch.set_num_threads(1)
 vis=os.environ.get('CUDA_VISIBLE_DEVICES','');
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=414000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{a.fold}.json')); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video']; run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json'; atomic(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu})
 for d in (marker.parent, latest.parent, metrics.parent): d.mkdir(parents=True, exist_ok=True)
 model=SafetyVectorBridge().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4); rng=np.random.default_rng(seed+11); steps=100 if a.smoke else a.steps; hist=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=np.asarray(vec(r['query_track_key'],meta,feats)); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key'); cks=sk+([hk] if hk in meta else []); cv=np.asarray([vec(k,meta,feats) for k in cks]); sv=np.asarray([vec(k,meta,feats) for k in sk]); ctx=sv.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8); qt=torch.tensor(q,device=dev).view(1,-1); ct=torch.tensor(cv,device=dev); raw=(ct*qt).sum(-1); sm=(torch.tensor(sv,device=dev)@ct.T).max(0).values; support_quality=float(sm.max().detach().cpu()); z,alpha,_=model(qt,torch.tensor(ctx,device=dev).view(1,-1),raw.max().view(1),torch.tensor([support_quality],device=dev),True); score=z@ct.T; labels=torch.tensor([1.]*len(sk)+[0.]*(len(cks)-len(sk)),device=dev); pos=score[0,:len(sk)].mean() if score.ndim==2 else score[:len(sk)].mean(); neg=score[0,len(sk):].max() if len(cks)>len(sk) else torch.tensor(0.,device=dev); loss=F.relu(0.08-pos+neg)+0.1*(score.squeeze(0)-raw).pow(2).mean()+0.02*alpha.pow(2).mean()
  opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
  if st%max(1,min(a.checkpoint_every,steps))==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'alpha_max':model.alpha_max,'protocol':'phase41_safety_constrained_vector_bridge','amp':'bf16' if dev.type=='cuda' else 'fp32'}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu()),'alpha':float(alpha.detach().mean().cpu())})
 atomic(metrics,{'fold':a.fold,'steps':steps,'history':hist,'checkpoint_best':str(best),'protocol':'phase41_safety_constrained_vector_bridge','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(done)}))
if __name__=='__main__': main()
