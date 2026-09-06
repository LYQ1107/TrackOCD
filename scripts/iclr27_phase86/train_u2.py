#!/usr/bin/env python3
"""Train the single registered Phase86 U2 relation route (CV or all-TRAIN)."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,tempfile,sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import set_context,stable_raw_topk
from src.iclr27_phase86.set_aware_relation import SetAwareResidualReranker
OUT=ROOT/'outputs/iclr27_phase86'; MAN=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_manifest.json'; DATA=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_features.npz'; CK=Path('/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def save_torch(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));os.close(fd)
 try:torch.save(v,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def load():
 z=np.load(DATA,allow_pickle=False); m=json.loads(MAN.read_text()); x=z['features'].astype(np.float32); off=z['offsets'].astype(np.int64); tar=z['targets'].astype(np.int64); cnt=z['candidate_counts'].astype(np.int64); meta=m['groups_meta'];ctx=[]
 for g in range(len(meta)):
  a,b=int(off[g]),int(off[g+1]);raw=x[a:b,15];hq=float(np.clip((float(x[a:b,16].mean())+1)/2,0,1)) if b>a else 0.;ctx.append(set_context(raw,int(cnt[g]),float(meta[g].get('source_length',0)),float(meta[g].get('source_variance',0)),hq))
 return m,x,off,tar,meta,np.asarray(ctx,np.float32)
def metrics(model,ids,x,off,tar,ctx,mean,std):
 raw1=rer1=raw5=rer5=rescue=harm=0; match=0
 for g in ids:
  a,b=int(off[g]),int(off[g+1]); n=b-a; target=int(tar[g]);
  if not n:continue
  raw=x[a:b,15]; order=stable_raw_topk(raw,n); X=torch.from_numpy((x[a:b]-mean)/std); d=model(X,torch.from_numpy(ctx[g])).detach().numpy(); score=raw+d; ro=stable_raw_topk(score,n)
  if target<n:
   match+=1; raw1+=int(order[0]==target);rer1+=int(ro[0]==target);raw5+=int(target in set(order[:5]));rer5+=int(target in set(ro[:5]));rescue+=int(order[0]!=target and ro[0]==target);harm+=int(order[0]==target and ro[0]!=target)
 return {'groups':len(ids),'match_groups':match,'raw_top1':raw1/max(1,match),'reranked_top1':rer1/max(1,match),'raw_top5':raw5/max(1,match),'reranked_top5':rer5/max(1,match),'rescue':rescue,'harm':harm,'net_rescue':rescue-harm}
def run(mode,fold,epochs,steps,balanced=False):
 torch.set_num_threads(4);torch.manual_seed(86200+fold);np.random.seed(86200+fold);m,x,off,tar,meta,ctx=load(); all_ids=list(range(len(meta))); 
 if mode=='cv': fit=[int(v) for v in m['folds'][str(fold)]['fit_groups']]; val=[int(v) for v in m['folds'][str(fold)]['validation_groups']]; tag=(f'u2_balanced_cv_f{fold}' if balanced else f'u2_cv_f{fold}'); ck=CK/f'{tag}.pt'; metric=OUT/'metrics'/f'{tag}.json'; marker=OUT/'completion'/f'{tag}.launched'; done=OUT/'completion'/f'{tag}.done'
 else: fit=all_ids;val=[];tag='u2_final_alltrain_v1';ck=CK/f'{tag}.pt';metric=OUT/'metrics'/f'{tag}.json';marker=OUT/'completion'/f'{tag}.launched';done=OUT/'completion'/f'{tag}.done'
 if done.exists():print(metric.read_text());return
 if marker.exists():raise RuntimeError(f'already launched {marker}')
 route_name='U2_SET_AWARE_RELATION_BALANCED' if balanced else 'U2_SET_AWARE_RELATION'; atom(marker,{'phase':86,'route':route_name,'mode':mode,'fold':fold,'pid':os.getpid(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'balanced_group_sampling':balanced})
 rows=np.concatenate([np.arange(off[g],off[g+1]) for g in fit]);mean=x[rows].mean(0).astype(np.float32);std=np.where(x[rows].std(0)<1e-5,1.,x[rows].std(0)).astype(np.float32);model=SetAwareResidualReranker();opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5);rng=np.random.default_rng(86200+fold);losses=[];steps_done=0
 match=[g for g in fit if int(tar[g])<int(off[g+1]-off[g])]
 for ep in range(epochs):
  if balanced:
   defer=[g for g in fit if int(tar[g])>=int(off[g+1]-off[g])]; max_bal=max(len(match),len(defer)); order=np.concatenate([rng.choice(match,max_bal,replace=True),rng.choice(defer,max_bal,replace=True)]).astype(int); rng.shuffle(order)
  else:
   order=list(fit);rng.shuffle(order)
  for g in order:
   a,b=int(off[g]),int(off[g+1]);n=b-a;X=torch.from_numpy((x[a:b]-mean)/std);c=torch.from_numpy(ctx[g]);raw=torch.from_numpy(x[a:b,15]);target=int(tar[g]);delta=model(X,c)
   if target<n:
    with torch.no_grad(): raw_ok=bool(int(torch.argmax(raw))==target)
    loss=F.cross_entropy((raw+delta).unsqueeze(0)/.1,torch.tensor([target]))*(4.0 if raw_ok else 1.0)
   else: loss=.1*torch.mean(delta**2)
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step();steps_done+=1;losses.append(float(loss.detach()))
   if steps and steps_done%steps==0:save_torch(CK/f'{tag}_step{steps_done:06d}.pt',{'model':model.state_dict(),'mean':mean,'std':std,'step':steps_done,'fold':fold,'route':'U2_SET_AWARE_RELATION','manifest_sha256':sha(MAN)})
 save_torch(ck,{'model':model.state_dict(),'mean':mean,'std':std,'steps':steps_done,'epochs':epochs,'fold':fold,'route':route_name,'manifest_sha256':sha(MAN),'seed':86200+fold,'balanced_group_sampling':balanced})
 out={'schema_version':'trackocd.phase86.u2_metrics.v1','phase':86,'route':route_name,'mode':mode,'fold':fold,'epochs':epochs,'steps':steps_done,'fit_groups':len(fit),'validation_groups':len(val),'fit_metrics':metrics(model,fit,x,off,tar,ctx,mean,std),'validation_metrics':metrics(model,val,x,off,tar,ctx,mean,std) if val else None,'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck),'manifest_sha256':sha(MAN),'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,'balanced_group_sampling':balanced,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom(metric,out);atom(done,{'status':'DONE','metrics':str(metric.resolve()),'checkpoint':str(ck.resolve()),'sha256':sha(ck)});print(json.dumps(out,indent=2,sort_keys=True))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['cv','final'],required=True);ap.add_argument('--fold',type=int,default=0);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--balanced',action='store_true');a=ap.parse_args();run(a.mode,a.fold,a.epochs,a.steps,a.balanced)
if __name__=='__main__':main()
