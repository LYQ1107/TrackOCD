#!/usr/bin/env python3
import argparse, json, os, random, tempfile
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from src.iclr27_phase19r.data.stream import Phase19RData
from scripts.iclr27_phase46.evaluate_controller import GatedData
from src.iclr27_phase47.correspondence import DomainAlignedEncoder
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase47'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def build_sets(data,fold):
 m=json.load(open(ROOT/'outputs/iclr27_phase47/manifests/fold_manifest.json')); fm=m['folds'][fold]; fitc=set(fm['fit_categories']); held=set(fm['held_categories']); fitv=set(fm['fit_videos']); valv=set(fm['validation_videos']); cat=data.track_category; vid=data.track_video
 fit={}; val={}
 for k,c in cat.items():
  if c in fitc and vid[k] in fitv: fit.setdefault(c,[]).append(k)
  if c in held and vid[k] in valv: val.setdefault(c,[]).append(k)
 fit={c:sorted(v) for c,v in fit.items() if len(set(vid[k] for k in v))>=2}; val={c:sorted(v) for c,v in val.items() if v}; return fit,val
def retrieval(model, vectors, val, device):
 keys=[k for c in sorted(val) for k in val[c]]; cats={k:c for c,ks in val.items() for k in ks}; vids={k:int(k.split(':')[0][1:]) for k in keys}; x=torch.tensor(np.asarray([vectors[k] for k in keys]),device=device); e=model(x).detach().cpu().numpy(); s=e@e.T; r1=[]; aps=[]; gaps=[]
 for i,k in enumerate(keys):
  cand=[j for j,z in enumerate(keys) if j!=i and vids[z]!=vids[k]]; pos=[j for j in cand if cats[keys[j]]==cats[k]]; neg=[j for j in cand if cats[keys[j]]!=cats[k]]
  if not pos or not neg: continue
  order=np.asarray(cand)[np.argsort(s[i,cand])[::-1]]; hit=np.asarray([int(j in pos) for j in order]); r1.append(float(hit[0])); c=np.cumsum(hit); aps.append(float(np.sum(c/(np.arange(len(hit))+1)*hit)/len(pos))); gaps.append(float(s[i,pos].max()-s[i,neg].max()))
 return {'r1':float(np.mean(r1)) if r1 else 0.,'map':float(np.mean(aps)) if aps else 0.,'hard_gap':float(np.mean(gaps)) if gaps else 0.,'queries':len(r1)}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--tag',default='phase47_formal_v1'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); a=ap.parse_args()
    torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
    dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); fold=a.fold; seed=47000+fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json'; atomic(marker,{'fold':fold,'pid':os.getpid(),'physical_gpu':a.expected_physical_gpu})
    Gate=__import__('src.iclr27_phase46.selective',fromlist=['ConditionalLogitGate']).ConditionalLogitGate; Bridge=__import__('src.iclr27_phase41.bridge',fromlist=['SafetyVectorBridge']).SafetyVectorBridge
    gate=Gate(); gate.load_state_dict(torch.load(ROOT/'outputs/iclr27_phase46/checkpoints'/f'phase46_formal_v1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge=Bridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model'])
    data=GatedData(fold,gate,bridge); fit,val=build_sets(data,fold); model=DomainAlignedEncoder().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); rng=np.random.default_rng(seed+9); cache={}
    def track_vec(k,p=16):
        key=(k,p)
        if key not in cache: cache[key]=data.prefix(k,min(p-1,len(data.track_rows[k])-1))[0]
        return cache[key]
    hist=[]; steps=2 if a.smoke else a.steps; cats=sorted(fit)
    for st in range(1,steps+1):
        c=int(rng.choice(cats)); ak=rng.choice(fit[c]); choices=[k for k in fit[c] if data.track_video[k]!=data.track_video[ak]] or fit[c]; p1=rng.choice(choices); p2=rng.choice([k for k in choices if k!=p1] or choices); nc=[x for x in cats if x!=c]; negk=rng.choice(fit[int(rng.choice(nc))]); short=int(rng.choice((1,2,4,8))); xa=torch.tensor(track_vec(ak),device=dev).view(1,-1); xp1=torch.tensor(track_vec(p1),device=dev).view(1,-1); xp2=torch.tensor(track_vec(p2),device=dev).view(1,-1); xn=torch.tensor(track_vec(negk),device=dev).view(1,-1); xs=torch.tensor(track_vec(ak,short),device=dev).view(1,-1); ea=model(xa); ep1=model(xp1); ep2=model(xp2); en=model(xn); es=model(xs); pos=.5*((ea*ep1).sum(-1)+(ea*ep2).sum(-1)); neg=(ea*en).sum(-1); loss=F.relu(.20-pos+neg).mean()+.10*(1-pos).mean()+.10*(1-(ea*es).sum(-1)).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step(); hist.append({'step':st,'loss':float(loss.detach().cpu()),'pos':float(pos.detach().cpu()),'neg':float(neg.detach().cpu())})
    vectors={k:track_vec(k,16) for c in val for k in val[c]}; v=retrieval(model,vectors,val,dev); payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':steps,'fold':fold,'seed':seed,'protocol':'phase47_domain_aligned_correspondence','validation':v}; torch.save(payload,latest); torch.save(payload,best); atomic(metrics,{'fold':fold,'steps':steps,'smoke':a.smoke,'fit_categories':cats,'validation':v,'history':hist,'checkpoint_best':str(best),'materialized_fit_tracks':sum(len(x) for x in fit.values()),'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(done,{'fold':fold,'steps':steps,'validation':v,'checkpoint':str(best)}); print(json.dumps({'fold':fold,'steps':steps,'validation':v},indent=2))
if __name__=='__main__': main()
