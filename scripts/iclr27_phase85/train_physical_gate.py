#!/usr/bin/env python3
"""TRAIN-only MLP64 selective reconnect gate for Phase85 temporal mean."""
from __future__ import annotations
import argparse,datetime as dt,hashlib,json,os,tempfile,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.physical_gate_model import PhysicalUnionGate
OUT=ROOT/'outputs/iclr27_phase85';DATA=OUT/'manifests/physical_gate_examples.npz';MAN=OUT/'manifests/physical_gate_examples.json';CK=Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def atom_torch(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));os.close(fd);torch.save(v,t);os.replace(t,p)
def evalm(model,x,y,mean,std,ids):
 if len(ids)==0:return {'rows':0}
 with torch.no_grad():p=torch.sigmoid(model(torch.from_numpy(((x[ids]-mean)/std).astype(np.float32)))).numpy()
 pred=p>=.5; yy=y[ids]>=.5; tp=int((pred&yy).sum());fp=int((pred&~yy).sum());fn=int((~pred&yy).sum());tn=int((~pred&~yy).sum());return {'rows':len(ids),'positive':int(yy.sum()),'negative':int((~yy).sum()),'pred_accept':int(pred.sum()),'accept_precision':tp/max(1,tp+fp),'accept_recall':tp/max(1,tp+fn),'false_reconnect_rate':fp/max(1,int((~yy).sum())),'keep_rate':tn/max(1,tn+fp),'accuracy':(tp+tn)/max(1,len(ids))}
def train(fold,tag,epochs,limit):
 z=np.load(DATA,allow_pickle=False);x=z['features'].astype(np.float32);y=z['labels'].astype(np.float32);b=z['fold'].astype(np.int64);fit=np.where(b!=fold)[0];val=np.where(b==fold)[0]; pos=[i for i in fit if y[i]>.5];neg=[i for i in fit if y[i]<.5]
 if not pos or not neg:raise RuntimeError('missing positive/negative fit examples')
 comp=OUT/'completion';met=OUT/'metrics';marker=comp/f'physical_gate_{tag}_f{fold}.launched';done=comp/f'physical_gate_{tag}_f{fold}.done';
 if done.exists():return json.loads((met/f'physical_gate_{tag}_f{fold}.json').read_text())
 if marker.exists():raise RuntimeError(f'already launched {marker}')
 atom(marker,{'phase':'Phase85','route':'SELECTIVE_PHYSICAL_UNION_GATE','tag':tag,'fold':fold,'pid':os.getpid(),'device':'cpu','created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 mean=x[fit].mean(0).astype(np.float32);std=np.where(x[fit].std(0)<1e-5,1.,x[fit].std(0)).astype(np.float32);m=PhysicalUnionGate(10,64);opt=torch.optim.Adam(m.parameters(),lr=2e-3,weight_decay=1e-5);rng=np.random.default_rng(85200+fold);steps=0;losses=[];n=max(len(pos),len(neg));
 while True:
  order=np.concatenate([rng.choice(pos,n,replace=True),rng.choice(neg,n,replace=True)]);rng.shuffle(order)
  for i in order:
   xx=torch.from_numpy(((x[i]-mean)/std).astype(np.float32)).unsqueeze(0); yy=torch.tensor([y[i]]);logit=m(xx);loss=F.binary_cross_entropy_with_logits(logit,yy);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.);opt.step();losses.append(float(loss.detach()));steps+=1
   if steps%1000==0:atom_torch(CK/f'physical_gate_{tag}_f{fold}_step{steps:06d}.pt',{'model':m.state_dict(),'mean':mean,'std':std,'step':steps,'fold':fold,'feature_dim':10,'manifest_sha256':sha(MAN)})
   if limit and steps>=limit:break
  if limit and steps>=limit:break
  if not limit and steps>=epochs*2*n:break
 cp=CK/f'physical_gate_{tag}_f{fold}_step{steps:06d}.pt';atom_torch(cp,{'model':m.state_dict(),'mean':mean,'std':std,'step':steps,'fold':fold,'feature_dim':10,'manifest_sha256':sha(MAN)});obj={'schema_version':'trackocd.phase85.physical_gate_metrics.v1','phase':'Phase85 P-selective','route':'SELECTIVE_PHYSICAL_UNION_GATE','tag':tag,'fold':fold,'epochs':epochs,'steps':steps,'feature_dim':10,'fit_metrics':evalm(m,x,y,mean,std,fit),'validation_metrics':evalm(m,x,y,mean,std,val),'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,'checkpoint':str(cp.resolve()),'checkpoint_sha256':sha(cp),'manifest':str(MAN.resolve()),'manifest_sha256':sha(MAN),'threshold':.5,'labels_posthoc_only':True,'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'device':'cpu'};atom(met/f'physical_gate_{tag}_f{fold}.json',obj);atom(done,{'status':'DONE','fold':fold,'tag':tag,'checkpoint':str(cp.resolve()),'metrics':str((met/f'physical_gate_{tag}_f{fold}.json').resolve())});return obj
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--steps',type=int,default=0);a=ap.parse_args();torch.set_num_threads(4);print(json.dumps(train(a.fold,a.tag,a.epochs,a.steps),indent=2,sort_keys=True))
if __name__=='__main__':main()
