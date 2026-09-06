#!/usr/bin/env python3
"""Phase86 U1: TRAIN-only OOF utility gate over frozen Phase85 experts.

The diagnostic and this gate are intentionally separate.  All expert scores
come from frozen Phase85 checkpoints; the U1 MLP sees only causal candidate
features and score summaries, never category/ID/text/GT fields.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, tempfile, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import set_context, stable_raw_topk
from src.iclr27_phase85.support_model import SupportReranker, numpy_predict

OUT=ROOT/'outputs/iclr27_phase86'; MAN=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_manifest.json'; DATA=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_features.npz'; CK85=Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints'); CK86=Path('/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints')

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w',encoding='utf-8') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def atom_torch(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));os.close(fd)
 try: torch.save(v,t);os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)

class UtilityGate(nn.Module):
 def __init__(self):
  super().__init__(); self.net=nn.Sequential(nn.Linear(64,32),nn.LayerNorm(32),nn.Tanh(),nn.Linear(32,1))
 def forward(self,x): return self.net(x).squeeze(-1)

def load_experts():
 m=json.loads(MAN.read_text()); z=np.load(DATA,allow_pickle=False); x=z['features'].astype(np.float32); off=z['offsets'].astype(np.int64); tar=z['targets'].astype(np.int64); cnt=z['candidate_counts'].astype(np.int64); meta=m['groups_meta']; ctx=[]
 for g in range(len(meta)):
  a,b=int(off[g]),int(off[g+1]); raw=x[a:b,15]; hq=float(np.clip((float(x[a:b,16].mean())+1.)/2.,0.,1.)) if b>a else 0.; ctx.append(set_context(raw,int(cnt[g]),float(meta[g].get('source_length',0)),float(meta[g].get('source_variance',0)),hq))
 experts={}
 for f in range(3):
  metric=json.loads((ROOT/f'outputs/iclr27_phase85/metrics/support_reranker_formal_r1_f{f}.json').read_text()); p=Path(metric['checkpoint']); zc=torch.load(p,map_location='cpu'); model=SupportReranker(19,10,64,.05); model.load_state_dict(zc['model']); model.eval(); experts[f]={'model':model,'mean':np.asarray(zc['mean'],np.float32),'std':np.asarray(zc['std'],np.float32),'checkpoint':str(p.resolve()),'sha256':sha(p)}
 return m,x,off,tar,cnt,meta,np.asarray(ctx,np.float32),experts

def feat_for_group(g,x,off,tar,meta,ctx,experts):
 a,b=int(off[g]),int(off[g+1]); xx=x[a:b]; raw=xx[:,15]; bridge=xx[:,16]; n=b-a; ef=experts[int(meta[g]['assigned_fold'])]['model']; ee=experts[int(meta[g]['assigned_fold'])]; rscore,p,delta=numpy_predict(ef,xx,ctx[g],raw,ee['mean'],ee['std'],'cpu'); ro=stable_raw_topk(raw,n); bo=stable_raw_topk(bridge,n); mo=stable_raw_topk(rscore,n)
 def margin(s,o): return float(s[o[0]]-s[o[1]]) if len(o)>1 else float(s[o[0]]) if len(o) else 0.
 def stats(s):
  if not len(s): return [0.]*8
  q=np.sort(np.asarray(s,np.float32))[::-1]; return [float(q[0]),float(q[1] if len(q)>1 else q[0]),float(q[2] if len(q)>2 else q[-1]),float(np.mean(q)),float(np.std(q)),float(np.max(q)-np.min(q)),float(np.percentile(q,25)),float(np.percentile(q,75))]
 # 10 context + means/stds over all 19 candidate fields + raw-top candidate first 16 fields = 64.
 f=np.concatenate([ctx[g],xx.mean(0),xx.std(0),xx[ro[0],:16] if len(ro) else np.zeros(16,np.float32)]).astype(np.float32)
 assert f.shape==(64,),f.shape
 target=int(tar[g]); raw_ok=bool(target<n and ro[0]==target); rerank_ok=bool(target<n and mo[0]==target); utility=1.0 if (not raw_ok and rerank_ok) else -4.0 if (raw_ok and not rerank_ok) else 0.0
 teacher=bool(float(meta[g].get('support_quality',0.))>=.2 and margin(bridge,bo)>=margin(raw,ro)+.005)
 extras={'group':g,'assigned_fold':int(meta[g]['assigned_fold']),'prefix':int(meta[g]['prefix']),'query_key':str(meta[g]['query_key']),'source_key':str(meta[g]['source_key']),'target_video':int(meta[g]['target_video']),'feature':f.tolist(),'utility':utility,'teacher_use':int(teacher),'raw_correct':int(raw_ok),'rerank_correct':int(rerank_ok),'raw_margin':margin(raw,ro),'bridge_margin':margin(bridge,bo),'rerank_margin':margin(rscore,mo),'support_quality':float(meta[g].get('support_quality',0.)),'rerank_score_probability':float(p),'candidate_count':int(n),'target_present':bool(target<n),'raw_top1':int(ro[0]) if len(ro) else None,'rerank_top1':int(mo[0]) if len(mo) else None,'bridge_top1':int(bo[0]) if len(bo) else None,'expert_checkpoint':experts[int(meta[g]['assigned_fold'])]['checkpoint']}
 return f,utility,teacher,extras

def build_table():
 m,x,off,tar,cnt,meta,ctx,experts=load_experts(); rows=[]
 for g in range(len(meta)):
  _,_,_,row=feat_for_group(g,x,off,tar,meta,ctx,experts); rows.append(row)
 return m,rows,experts

def expert_ledger(experts):
 return {str(f):{'checkpoint':v['checkpoint'],'sha256':v['sha256']} for f,v in experts.items()}

def train_one(train, val, seed, steps):
 rng=np.random.default_rng(seed); model=UtilityGate(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5); X=np.asarray([r['feature'] for r in train],np.float32); y=np.asarray([r['utility'] for r in train],np.float32); mean=X.mean(0); std=np.where(X.std(0)<1e-5,1.,X.std(0)); Xt=torch.from_numpy((X-mean)/std); yt=torch.from_numpy(y); losses=[]
 for s in range(steps):
  idx=rng.integers(0,len(train),size=min(256,len(train))); pred=model(Xt[idx]); loss=torch.nn.functional.smooth_l1_loss(pred,yt[idx]); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step(); losses.append(float(loss.detach()))
 return model,mean,std,losses

def eval_gate(model,mean,std,rows):
 utility=[]; rescue=0; harm=0; neutral=0; teacher=[]; use=[]
 for r in rows:
  with torch.no_grad(): pred=float(model(torch.from_numpy(((np.asarray(r['feature'],np.float32)-mean)/std)[None,:]))[0])
  take=pred>0.; u=float(r['utility']); utility.append(pred); use.append(int(take)); teacher.append(int(r['teacher_use']))
  if take and u>0:rescue+=1
  elif take and u<0:harm+=1
  else:neutral+=1
 return {'groups':len(rows),'predicted_use':int(sum(use)),'bridge_use_rate':float(np.mean(use)) if use else 0.,'rescue':rescue,'harm':harm,'neutral_or_no_use':neutral,'net_rescue':rescue-harm,'teacher_use_rate':float(np.mean(teacher)) if teacher else 0.,'teacher_agreement':float(np.mean(np.asarray(use)==np.asarray(teacher))) if teacher else 0.,'predicted_utility_mean':float(np.mean(utility)) if utility else 0.}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=('smoke','targeted','formal'),required=True);ap.add_argument('--tag',required=True);ap.add_argument('--fold',type=int,default=0);a=ap.parse_args(); torch.set_num_threads(4)
 m,rows,experts=build_table(); by={f:[r for r in rows if int(r['assigned_fold'])==f] for f in range(3)}
 # The expert predictions are OOF: each row is scored by the frozen Phase85 checkpoint whose validation fold owns it.
 if a.mode=='smoke': train=rows[:min(256,len(rows))]; val=rows[:min(128,len(rows))]; steps=100; name=f'u1_{a.tag}_smoke_f{a.fold}'
 elif a.mode=='targeted': train=[r for f in range(3) if f!=a.fold for r in by[f]]; val=by[a.fold]; steps=500; name=f'u1_{a.tag}_targeted_f{a.fold}'
 else: train=[r for f in range(3) if f!=a.fold for r in by[f]]; val=by[a.fold]; steps=2000; name=f'u1_{a.tag}_formal_f{a.fold}'
 comp=OUT/'completion'; done=comp/(name+'.done'); launched=comp/(name+'.launched'); ck=CK86/(name+'.pt'); met=OUT/'metrics'/(name+'.json')
 if done.exists(): print(met.read_text()); return
 if launched.exists(): raise RuntimeError(f'unit already launched without done: {launched}')
 atom(launched,{'phase':86,'route':'U1_OOF_UTILITY_GATE','mode':a.mode,'fold':a.fold,'tag':a.tag,'pid':os.getpid(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 model,mean,std,losses=train_one(train,val,86000+a.fold,steps); tr=eval_gate(model,mean,std,train); va=eval_gate(model,mean,std,val); ledger=expert_ledger(experts); atom_torch(ck,{'model':model.state_dict(),'mean':mean,'std':std,'steps':steps,'fold':a.fold,'input_dim':64,'hidden_dim':32,'expert_checkpoints':ledger,'manifest_sha256':sha(MAN),'route':'U1_OOF_UTILITY_GATE'})
 out={'schema_version':'trackocd.phase86.u1_metrics.v1','phase':86,'route':'U1_OOF_UTILITY_GATE','tag':a.tag,'mode':a.mode,'fold':a.fold,'steps':steps,'input_dim':64,'hidden_dim':32,'train_groups':len(train),'validation_groups':len(val),'train_metrics':tr,'validation_metrics':va,'loss_first':losses[0],'loss_last':losses[-1],'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck),'manifest':str(MAN.resolve()),'manifest_sha256':sha(MAN),'expert_checkpoints':ledger,'labels':{'help':1,'harm':-4,'both_correct_or_wrong':0,'selection':'predicted utility > 0 => use reranker else exact raw'},'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'gt_fields_in_feature_tensor':False}
 atom(met,out); atom(done,{'status':'DONE','metrics':str(met.resolve()),'checkpoint':str(ck.resolve()),'sha256':sha(met)}); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__':main()
