#!/usr/bin/env python3
import argparse,json,os,tempfile,random
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase43.selective import PolicyDistilledGate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase43'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p=16):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--checkpoint-every',type=int,default=500); ap.add_argument('--tag',default='policy_formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args(); torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','');
 if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=434000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{a.fold}.json')); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video']; run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json';
 for d in (marker.parent,latest.parent,metrics.parent): d.mkdir(parents=True,exist_ok=True)
 atomic(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu}); bridge=SafetyVectorBridge().to(dev); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{a.fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); gate=PolicyDistilledGate().to(dev); opt=torch.optim.AdamW(gate.parameters(),lr=3e-4); rng=np.random.default_rng(seed+9); steps=100 if a.smoke else a.steps; hist=[]
 for st in range(1,steps+1):
  r=fit[int(rng.integers(len(fit)))]; q=np.asarray(vec(r['query_track_key'],meta,feats)); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key'); cks=sk+([hk] if hk in meta else []); cv=np.asarray([vec(k,meta,feats) for k in cks]); qt=torch.tensor(q,device=dev).view(1,-1); ct=torch.tensor(cv,device=dev); raw=(qt*ct).sum(-1); sv=torch.tensor(np.asarray([vec(k,meta,feats) for k in sk]),device=dev); ctx=sv.mean(0); ctx/=ctx.norm().clamp_min(1e-8); sq=(sv@qt.squeeze(0)).max(); z,alpha,_=bridge(qt,ctx.view(1,-1),raw.max().view(1),sq.view(1),True); bs=(z@ct.T).squeeze(0); rm=raw[:len(sk)].max()-raw[len(sk):].max(); bm=bs[:len(sk)].max()-bs[len(sk):].max(); teacher=((sq>=0.2)&(bm>=rm+0.005)).float().view(1); p=gate(rm.view(1),bm.view(1),sq.view(1),alpha.view(1),torch.zeros(1,device=dev)); unsafe=bool(raw.argmax()<len(sk) and bs.argmax()>=len(sk)); loss=F.binary_cross_entropy(p,teacher)+(0.15*p if unsafe else 0.0)+0.005*(p*(1-p)).mean()
  opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(),2); opt.step()
  if st%max(1,min(a.checkpoint_every,steps))==0 or st==steps:
   payload={'model':gate.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase43_train_only_policy_distilled_gate','amp':'bf16' if dev.type=='cuda' else 'fp32'}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu()),'p_bridge':float(p.detach().cpu()),'teacher':float(teacher.item())})
 atomic(metrics,{'fold':a.fold,'steps':steps,'history':hist,'checkpoint_best':str(best),'protocol':'phase43_train_only_policy_distilled_gate','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'done':str(done)}))
if __name__=='__main__': main()
