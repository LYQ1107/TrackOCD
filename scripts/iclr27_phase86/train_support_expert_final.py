#!/usr/bin/env python3
"""All-TRAIN deployment fit for the already validated Phase85 support expert."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, tempfile, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import set_context, stable_raw_topk
from src.iclr27_phase85.support_model import SupportReranker
OUT=ROOT/'outputs/iclr27_phase86'; MAN=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_manifest.json'; DATA=ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_features.npz'; CK=Path('/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def atom_torch(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent)); os.close(fd)
 try: torch.save(v,t); os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def load():
 z=np.load(DATA,allow_pickle=False); m=json.loads(MAN.read_text()); x=z['features'].astype(np.float32); off=z['offsets'].astype(np.int64); tar=z['targets'].astype(np.int64); cnt=z['candidate_counts'].astype(np.int64); meta=m['groups_meta']; ctx=[]
 for g in range(len(meta)):
  a,b=int(off[g]),int(off[g+1]); raw=x[a:b,15]; hq=float(np.clip((float(x[a:b,16].mean())+1)/2,0,1)) if b>a else 0.; ctx.append(set_context(raw,int(cnt[g]),float(meta[g].get('source_length',0)),float(meta[g].get('source_variance',0)),hq))
 return m,x,off,tar,cnt,meta,np.asarray(ctx,np.float32)
def main():
 torch.set_num_threads(4); torch.manual_seed(86001); np.random.seed(86001)
 tag='final_alltrain_v1'; comp=OUT/'completion'; launched=comp/f'support_expert_{tag}.launched'; done=comp/f'support_expert_{tag}.done'; metric=OUT/'metrics'/f'support_expert_{tag}.json'; ck=CK/f'support_expert_{tag}.pt'
 if done.exists(): print(metric.read_text()); return
 if launched.exists(): raise RuntimeError(f'already launched: {launched}')
 atom_json(launched,{'phase':86,'route':'U1_FINAL_SUPPORT_EXPERT','tag':tag,'pid':os.getpid(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 m,x,off,tar,cnt,meta,ctx=load(); ids=list(range(len(meta))); rows=[]
 for g in ids:
  a,b=int(off[g]),int(off[g+1]); rows.extend(range(a,b))
 mean=x[np.asarray(rows)].mean(0).astype(np.float32); std=np.where(x[np.asarray(rows)].std(0)<1e-5,1.,x[np.asarray(rows)].std(0)).astype(np.float32)
 match=[g for g in ids if int(tar[g]) < int(off[g+1]-off[g])]; defer=[g for g in ids if int(tar[g]) >= int(off[g+1]-off[g])]
 model=SupportReranker(19,10,64,.05); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5); rng=np.random.default_rng(86001); losses=[]; steps=0; epochs=15; max_bal=max(len(match),len(defer))
 for ep in range(epochs):
  order=np.concatenate([rng.choice(match,max_bal,replace=True),rng.choice(defer,max_bal,replace=True)]); rng.shuffle(order)
  for g in order:
   a,b=int(off[g]),int(off[g+1]); n=b-a; X=torch.from_numpy((x[a:b]-mean)/std); c=torch.from_numpy(ctx[g]); raw=torch.from_numpy(x[a:b,15]); target=int(tar[g]); delta,dl=model(X,c); loss=F.binary_cross_entropy_with_logits(dl,torch.tensor(float(target>=n)))
   if target<n: loss=loss+F.cross_entropy((raw+delta).unsqueeze(0)/.1,torch.tensor([target]))
   opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); steps+=1; losses.append(float(loss.detach()))
  atom_torch(CK/f'{tag}_epoch{ep+1:02d}.pt',{'model':model.state_dict(),'mean':mean,'std':std,'epoch':ep+1,'steps':steps,'architecture':{'candidate_dim':19,'context_dim':10,'hidden':64,'residual_scale':.05},'manifest_sha256':sha(MAN),'data_sha256':sha(DATA),'seed':86001,'training_source':'all_non_event_train'})
 atom_torch(ck,{'model':model.state_dict(),'mean':mean,'std':std,'epochs':epochs,'steps':steps,'architecture':{'candidate_dim':19,'context_dim':10,'hidden':64,'residual_scale':.05},'training_groups':len(ids),'match_groups':len(match),'defer_groups':len(defer),'training_source':'all_non_event_train','manifest_sha256':sha(MAN),'data_sha256':sha(DATA),'seed':86001})
 out={'schema_version':'trackocd.phase86.final_support_expert.v1','phase':86,'route':'U1_FINAL_SUPPORT_EXPERT','tag':tag,'epochs':epochs,'steps':steps,'groups':len(ids),'match_groups':len(match),'defer_groups':len(defer),'mean':mean.tolist(),'std':std.tolist(),'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck),'manifest':str(MAN.resolve()),'manifest_sha256':sha(MAN),'data':str(DATA.resolve()),'data_sha256':sha(DATA),'architecture':{'candidate_dim':19,'context_dim':10,'hidden':64,'residual_scale':.05},'seed':86001,'loss_first':losses[0],'loss_last':losses[-1],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom_json(metric,out); atom_json(done,{'status':'DONE','metrics':str(metric.resolve()),'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck)}); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
