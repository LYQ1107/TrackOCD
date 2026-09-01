#!/usr/bin/env python3
import argparse,json,os,tempfile,random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase31.reranker import MonotonicRawReranker
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase31'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def track_vec(k,meta,feats):
 x=feats[np.asarray(meta[k]['rows'])[:16]]; v=x.mean(0); v/=max(np.linalg.norm(v),1e-8); return v
def pair(a,b,meta,feats):
 va,vb=track_vec(a,meta,feats),track_vec(b,meta,feats); raw=float(np.dot(va,vb)); ma,mb=meta[a],meta[b]
 vals=[raw,abs(ma['area']-mb['area']),abs(np.log(max(ma['area'],1e-6))-np.log(max(mb['area'],1e-6))),abs(ma['length']-mb['length'])/16, min(ma['length'],mb['length'])/16, 1.0 if ma['video']==mb['video'] else 0.0, float(ma['category']==mb['category']), 0.0,0.0,0.0]
 return raw,np.asarray(vals,np.float32)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--steps',type=int,default=1000); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--checkpoint-every',type=int,default=500); ap.add_argument('--tag',default='reranker_formal'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--expected-physical-gpu',type=int,default=-1); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
 torch.set_num_threads(1); vis=os.environ.get('CUDA_VISIBLE_DEVICES','');
 if args.expected_physical_gpu>=0 and vis and vis.split(',')[0].strip()!=str(args.expected_physical_gpu): raise RuntimeError('GPU mapping mismatch')
 dev=torch.device(args.device if torch.cuda.is_available() else 'cpu'); seed=313100+args.fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 rows,tracks,feats=load_tracks(); meta=track_metadata(rows,tracks); man=json.load(open(OUT.parent / ('iclr27_phase30/manifests/episode_manifest_f%d.json' % args.fold))); fit=[r for r in man['records'] if r['split']=='fit']; val=[r for r in man['records'] if r['split']=='val']; pos=[r for r in fit if r['kind']=='multi_positive_cross_video'];
 pairs=[]
 for r in pos:
  q=r['query_track_key'];
  for s in r.get('support_track_keys',[]):
   if q in meta and s in meta: pairs.append((q,s,1))
  h=r.get('hard_negative_track_key');
  if q in meta and h in meta: pairs.append((q,h,0))
 run=f"{args.tag}_{'smoke_' if args.smoke else ''}f{args.fold}"; marker=OUT/'completion'/f'{run}.launched'; done=OUT/'completion'/f'{run}.done'; latest=OUT/'checkpoints'/f'{run}_latest.pt'; best=OUT/'checkpoints'/f'{run}_best.pt'; metrics=OUT/'metrics'/f'{run}.json'
 if marker.exists(): raise RuntimeError(f'marker exists {marker}')
 atomic(marker,{'fold':args.fold,'pid':os.getpid(),'gpu':args.expected_physical_gpu})
 model=MonotonicRawReranker().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); rng=np.random.default_rng(seed+7); steps=100 if args.smoke else args.steps; hist=[]; bestscore=-1e9
 for st in range(1,steps+1):
  batch=[pairs[int(rng.integers(0,len(pairs)))] for _ in range(min(args.batch_size,len(pairs)))]; raws=[]; metas=[]; labels=[]
  for a,b,y in batch: ra,fa=pair(a,b,meta,feats); raws.append(ra); metas.append(fa); labels.append(y)
  rawt=torch.tensor(raws,device=dev); mt=torch.tensor(np.asarray(metas),device=dev); y=torch.tensor(labels,dtype=torch.float32,device=dev); opt.zero_grad(); sc=model(rawt,mt); loss=F.binary_cross_entropy_with_logits(5*sc,y*0.8+0.1)+0.01*model.residual[3].weight.pow(2).mean();
  if not torch.isfinite(loss): raise RuntimeError('nonfinite')
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
  if st%max(1,min(args.checkpoint_every,steps))==0 or st==steps:
   payload={'model':model.state_dict(),'optimizer':opt.state_dict(),'step':st,'seed':seed,'protocol':'phase31_raw_space_monotonic_reranker','sealed_inputs_not_read':['DEV+','Q1','public new-model labels','future','IDs/text/held GT']}; torch.save(payload,latest); torch.save(payload,best); hist.append({'step':st,'loss':float(loss.detach().cpu()),'residual_bound':float((0.2*torch.sigmoid(model.scale)).detach().cpu())})
 atomic(metrics,{'fold':args.fold,'steps':steps,'pairs':len(pairs),'history':hist,'checkpoint_best':str(best),'protocol':'phase31_raw_space_monotonic_reranker'}); atomic(done,{'fold':args.fold,'steps':steps,'checkpoint':str(best)})
 print(json.dumps({'fold':args.fold,'steps':steps,'pairs':len(pairs),'done':str(done)},indent=2))
if __name__=='__main__': main()
