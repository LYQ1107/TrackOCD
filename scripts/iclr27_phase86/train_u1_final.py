#!/usr/bin/env python3
"""Fit the fixed U1 utility gate on all OOF TRAIN meta rows."""
from __future__ import annotations
import datetime as dt, hashlib, json, os, tempfile, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.iclr27_phase86.run_u1_selective import UtilityGate, build_table
OUT=ROOT/'outputs/iclr27_phase86'; CK=Path('/data2/usr_for_deadline/trackocd_phase86/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 try:
  with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
  os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def save_torch(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent)); os.close(fd)
 try: torch.save(v,t); os.replace(t,p)
 finally:
  if os.path.exists(t): os.unlink(t)
def main():
 torch.set_num_threads(4); torch.manual_seed(86100); np.random.seed(86100); tag='final_alltrain_v1_fix1'; comp=OUT/'completion'; launched=comp/f'u1_{tag}.launched'; done=comp/f'u1_{tag}.done'; metric=OUT/'metrics'/f'u1_{tag}.json'; ck=CK/'u1_final_alltrain_v1.pt'
 if done.exists(): print(metric.read_text()); return
 if launched.exists(): raise RuntimeError(f'already launched: {launched}')
 atom(launched,{'phase':86,'route':'U1_FINAL_UTILITY_GATE','tag':tag,'pid':os.getpid(),'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 manifest,rows,experts=build_table(); X=np.asarray([r['feature'] for r in rows],np.float32); y=np.asarray([r['utility'] for r in rows],np.float32); mean=X.mean(0).astype(np.float32); std=np.where(X.std(0)<1e-5,1.,X.std(0)).astype(np.float32); Xt=torch.from_numpy((X-mean)/std); yt=torch.from_numpy(y); model=UtilityGate(); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-5); rng=np.random.default_rng(86100); losses=[]
 labels={k:int(sum(1 for r in rows if (r['utility']==v))) for k,v in [('HELP',1.0),('HARM',-4.0),('BOTH_CORRECT',0.0)]}; labels['BOTH_WRONG']=int(sum(1 for r in rows if r['utility']==0.0 and not r['raw_correct'] and not r['rerank_correct']))
 atom(OUT/'audit/u1_oof_label_distribution.json',{'schema_version':'trackocd.phase86.u1_oof_labels.v1','rows':len(rows),'counts':labels,'definitions':{'HELP':'+1','HARM':'-4','BOTH':'0'},'manifest_rows':len(rows),'public_dev_q1_sealed_accessed':False})
 for step in range(2000):
  idx=rng.integers(0,len(rows),size=min(256,len(rows))); pred=model(Xt[idx]); loss=torch.nn.functional.smooth_l1_loss(pred,yt[idx]); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step(); losses.append(float(loss.detach()))
 save_torch(ck,{'model':model.state_dict(),'mean':mean,'std':std,'steps':2000,'input_dim':64,'hidden_dim':32,'training_source':'all_OOF_meta_rows','meta_rows':len(rows),'phase85_expert_hashes':{str(f):v['sha256'] for f,v in experts.items()},'manifest_sha256':sha(ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_manifest.json'),'seed':86100})
 out={'schema_version':'trackocd.phase86.final_u1.v1','phase':86,'route':'U1_FINAL_UTILITY_GATE','tag':tag,'steps':2000,'input_dim':64,'hidden_dim':32,'training_source':'all_OOF_meta_rows','meta_rows':len(rows),'label_distribution':labels,'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck),'manifest_sha256':sha(ROOT/'outputs/iclr27_phase85/manifests/phase85_support_prefix_manifest.json'),'phase85_expert_hashes':{str(f):v['sha256'] for f,v in experts.items()},'seed':86100,'loss_first':losses[0],'loss_last':losses[-1],'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False}
 atom(metric,out); atom(done,{'status':'DONE','metrics':str(metric.resolve()),'checkpoint':str(ck.resolve()),'checkpoint_sha256':sha(ck)}); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
