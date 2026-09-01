#!/usr/bin/env python3
import argparse, json, os, random, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase46.selective import ConditionalLogitGate

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT/'outputs/iclr27_phase46'; PREFIXES=(1,2,4,8,16)

def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True); fd,tmp=tempfile.mkstemp(dir=str(path.parent), prefix='.'+path.name)
    with os.fdopen(fd,'w') as f:
        json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def vec(key, meta, feats, prefix):
    x=feats[np.asarray(meta[key]['rows'])[:min(prefix,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)

def materialize(fold, dev):
    rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr)
    man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json'))
    fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video']
    bridge=SafetyVectorBridge().to(dev); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval()
    feats_out=[]; labels=[]; unsafe=[]
    with torch.no_grad():
      for r in fit:
        qk=r.get('query_track_key'); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key')
        if qk not in meta or not sk or hk not in meta: continue
        for pfx in PREFIXES:
          q=np.asarray(vec(qk,meta,feats,pfx)); sv=np.asarray([vec(k,meta,feats,pfx) for k in sk]); h=np.asarray(vec(hk,meta,feats,pfx)); c=np.concatenate([sv,[h]],0)
          raw=q@c.T; ctx=sv.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8)
          qt=torch.tensor(q,device=dev).view(1,-1); ct=torch.tensor(c,device=dev); st=torch.tensor(sv,device=dev)
          sq=(st@qt.squeeze(0)).max(); z,alpha,_=bridge(qt,torch.tensor(ctx,device=dev).view(1,-1),torch.tensor([float(raw.max())],device=dev),sq.view(1),True); bs=(z@ct.T).squeeze(0).cpu().numpy()
          rm=float(raw[:len(sk)].max()-raw[len(sk):].max()); bm=float(bs[:len(sk)].max()-bs[len(sk):].max()); teacher=float(sq.item()>=.2 and bm>=rm+.005); bad=bool(raw.argmax()<len(sk) and bs.argmax()>=len(sk))
          feats_out.append([rm,bm,float(sq.item()),float(alpha.item()),0.0]); labels.append(teacher); unsafe.append(float(bad))
    x=torch.tensor(np.asarray(feats_out),dtype=torch.float32,device=dev); y=torch.tensor(labels,dtype=torch.float32,device=dev); u=torch.tensor(unsafe,dtype=torch.float32,device=dev)
    return x,y,u

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--tag',default='formal'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); ap.add_argument('--checkpoint-every',type=int,default=500); a=ap.parse_args()
    torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if a.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(a.expected_physical_gpu): raise RuntimeError('GPU mismatch')
    dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); seed=46000+a.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    run=f"{a.tag}_{'smoke_' if a.smoke else ''}f{a.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json'; atomic(marker,{'fold':a.fold,'pid':os.getpid(),'gpu':a.expected_physical_gpu,'materialized':True})
    x,y,u=materialize(a.fold,dev); npos=float(y.sum().item()); nneg=float(len(y)-npos); pw=nneg/max(npos,1.0); gate=ConditionalLogitGate().to(dev); opt=torch.optim.AdamW(gate.parameters(),lr=3e-4); steps=100 if a.smoke else int(a.steps); hist=[]
    for st in range(1,steps+1):
        logits=gate(x[:,0],x[:,1],x[:,2],x[:,3],x[:,4]); bce=F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(pw,device=dev)); signed=2*y-1; margin_pen=F.relu(0.5-signed*logits).mean(); neg_cond=((x[:,2]<.2)|(x[:,1]-x[:,0]<.005)).float(); cond_pen=(neg_cond*F.relu(logits)).mean(); safety=(u*F.relu(logits)).mean(); loss=bce+0.05*margin_pen+0.10*cond_pen+0.50*safety
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(gate.parameters(),2.0); opt.step()
        if st%max(1,min(a.checkpoint_every,steps))==0 or st==steps:
            payload={'model':gate.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'pos_weight':pw,'materialized_examples':int(len(y)),'protocol':'phase46_full_materialized_balanced_bce_conditioned_gate','amp':'bf16' if dev.type=='cuda' else 'fp32'}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu()),'bce':float(bce.detach().cpu()),'margin_pen':float(margin_pen.detach().cpu()),'condition_pen':float(cond_pen.detach().cpu()),'safety_pen':float(safety.detach().cpu()),'p_mean':float(torch.sigmoid(logits).mean().detach().cpu()),'p_lt_0.5':float((torch.sigmoid(logits)<.5).float().mean().detach().cpu())})
    atomic(metrics,{'fold':a.fold,'steps':steps,'materialized_examples':int(len(y)),'teacher_positive_rate':float(y.mean().cpu()),'teacher_negative_rate':float(1-y.mean().cpu()),'history':hist,'checkpoint_best':str(best),'protocol':'phase46_full_materialized_balanced_bce_conditioned_gate','sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(done,{'fold':a.fold,'steps':steps,'checkpoint':str(best)}); print(json.dumps({'fold':a.fold,'steps':steps,'examples':len(y),'done':str(done)}))

if __name__=='__main__': main()
