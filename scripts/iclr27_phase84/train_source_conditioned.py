#!/usr/bin/env python3
"""Train the single B84S listwise source-conditioned selector (NumPy)."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, tempfile
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase84'; DATA=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/manifests/b84s_candidate_features.npz'); MAN=OUT/'manifests/b84s_native_manifest.json'
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,n=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));
 try:
  with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(n,p)
 finally:
  if os.path.exists(n):os.unlink(n)
def atom_npz(p,**kw):
 p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name('.'+p.name+'.tmp.npz'); np.savez(tmp,**kw); os.replace(tmp,p)
def evaluate(w,b,mean,std,x,offsets,targets,groups):
 rows=[]; loss=[]
 for g in groups:
  a,z=int(offsets[g]),int(offsets[g+1]); X=(x[a:z]-mean)/std; n=z-a; logits=np.concatenate([X@w,np.asarray([b])]); logits-=logits.max(); prob=np.exp(logits);prob/=max(float(prob.sum()),1e-12); choice=int(np.argmax(prob)); target=int(targets[g]); target=min(target,n); loss.append(float(-np.log(max(float(prob[target]),1e-12)))); rows.append((choice,target,n))
 rel=[r for r in rows if r[1]<r[2]]; cand1=sum(int(r[0]==r[1]) for r in rel); cand5=0
 for g,(choice,target,n) in zip(groups,rows):
  a,z=int(offsets[g]),int(offsets[g+1]); X=(x[a:z]-mean)/std; logits=X@w; order=np.argsort(logits)[::-1]
  if target<n and target in set(order[:5]): cand5+=1
 defer=[r for r in rows if r[1]>=r[2]]; defer_correct=sum(int(r[0]>=r[2]) for r in defer); all_correct=sum(int(r[0]==r[1]) for r in rows)
 return {'groups':len(rows),'target_candidate_groups':len(rel),'target_defer_groups':len(defer),'candidate_top1_recall':cand1/max(1,len(rel)),'candidate_top5_recall':cand5/max(1,len(rel)),'defer_recall':defer_correct/max(1,len(defer)),'candidate_or_defer_accuracy':all_correct/max(1,len(rows)),'predicted_defer_groups':sum(int(r[0]>=r[2]) for r in rows),'mean_nll':float(np.mean(loss)) if loss else 0.0}
def train_fold(fold,tag,epochs,steps_limit=0,smoke=False):
 z=np.load(DATA,allow_pickle=False); x=z['features'].astype(np.float32); offsets=z['offsets'].astype(np.int64); targets=z['targets'].astype(np.int64); m=json.loads(MAN.read_text()); fs=m['folds'][str(fold)]; fit=[int(g) for g in fs['fit_groups']]; val=[int(g) for g in fs['validation_groups']];
 if steps_limit: fit=fit[:max(1,min(len(fit),64))]
 fit_rows=np.concatenate([np.arange(offsets[g],offsets[g+1]) for g in fit]) if fit else np.arange(len(x)); mean=x[fit_rows].mean(0); std=x[fit_rows].std(0); std=np.where(std<1e-5,1.,std).astype(np.float32); w=np.zeros(x.shape[1],np.float32); b=0.; rng=np.random.default_rng(84084+fold); losses=[]; steps=0; ckdir=Path('/data2/usr_for_deadline/trackocd_phase84/project_outputs/checkpoints'); comp=OUT/'completion'; met=OUT/'metrics'; ckdir.mkdir(parents=True,exist_ok=True);comp.mkdir(parents=True,exist_ok=True);met.mkdir(parents=True,exist_ok=True); marker=comp/f'b84s_{tag}_f{fold}.launched'; done=comp/f'b84s_{tag}_f{fold}.done';
 if done.exists(): return json.loads((met/f'b84s_{tag}_f{fold}.json').read_text())
 if marker.exists(): raise RuntimeError(f'unit already launched without done: {marker}')
 atom_json(marker,{'phase':'Phase84','route':'B84S','tag':tag,'fold':fold,'pid':os.getpid(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'gpu':'cpu','epochs':epochs})
 order=np.asarray(fit,np.int64)
 for epoch in range(epochs):
  rng.shuffle(order)
  for g in order:
   a,z0=int(offsets[g]),int(offsets[g+1]); X=(x[a:z0]-mean)/std; n=z0-a; logits=np.concatenate([X@w,np.asarray([b])]); logits-=logits.max(); probs=np.exp(logits);probs/=max(float(probs.sum()),1e-12); target=min(int(targets[g]),n); loss=-np.log(max(float(probs[target]),1e-12)); grad=probs[:-1].astype(np.float32)
   if target < n: grad[target] -= 1.0
   w-=np.float32(0.04)*((X.T@grad)/max(1,n)); b-=np.float32(0.04)*(float(probs[-1])-(1. if target==n else 0.)); losses.append(float(loss));steps+=1
   if steps%1000==0: atom_npz(ckdir/f'b84s_{tag}_f{fold}_step{steps:06d}.npz',w=w,b=np.asarray([b],np.float32),mean=mean,std=std,step=np.asarray([steps]),fold=np.asarray([fold]));
   if steps_limit and steps>=steps_limit: break
  if steps_limit and steps>=steps_limit: break
 cp=ckdir/f'b84s_{tag}_f{fold}_step{steps:06d}.npz'; atom_npz(cp,w=w,b=np.asarray([b],np.float32),mean=mean,std=std,step=np.asarray([steps]),fold=np.asarray([fold])); fit_m=evaluate(w,b,mean,std,x,offsets,targets,fit); val_m=evaluate(w,b,mean,std,x,offsets,targets,val); obj={'schema_version':'trackocd.phase84.b84s_selector_metrics.v1','phase':'Phase84 B84S','route':'SOURCE_CONDITIONED_LISTWISE','tag':tag,'fold':fold,'epochs':epochs,'steps':steps,'fit_groups':len(fit),'validation_groups':len(val),'feature_dim':int(x.shape[1]),'fit_metrics':fit_m,'validation_metrics':val_m,'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,'checkpoint':str(cp.resolve()),'checkpoint_sha256':sha(cp),'manifest':str(MAN.resolve()),'manifest_sha256':sha(MAN),'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'gt_fields_in_feature_tensor':False,'gpu':'cpu'}; atom_json(met/f'b84s_{tag}_f{fold}.json',obj); atom_json(done,{'status':'DONE','fold':fold,'tag':tag,'metrics':str((met/f'b84s_{tag}_f{fold}.json').resolve()),'checkpoint':str(cp.resolve())}); return obj
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--tag',default='formal');ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--steps',type=int,default=0);a=ap.parse_args(); result=train_fold(a.fold,a.tag,a.epochs,a.steps,bool(a.steps)); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
