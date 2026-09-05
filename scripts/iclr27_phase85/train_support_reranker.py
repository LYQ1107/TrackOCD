#!/usr/bin/env python3
"""Train the single Phase85 raw-anchored set reranker and defer gate."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, tempfile, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.iclr27_phase85.raw_candidate_anchor import set_context,stable_raw_topk
from src.iclr27_phase85.support_model import SupportReranker,numpy_predict
OUT=ROOT/'outputs/iclr27_phase85'; DATA=OUT/'manifests/phase85_support_prefix_features.npz'; MAN=OUT/'manifests/phase85_support_prefix_manifest.json'; CFG=ROOT/'configs/iclr27_phase85/support_v1.json'; CK=Path('/data2/usr_for_deadline/trackocd_phase85/project_outputs/checkpoints')
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atom_json(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent))
 with os.fdopen(fd,'w') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def atom_torch(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);fd,t=tempfile.mkstemp(prefix='.'+p.name+'.',dir=str(p.parent));os.close(fd)
 try: torch.save(obj,t); os.replace(t,p)
 finally:
  if os.path.exists(t):os.unlink(t)
def load_data():
 z=np.load(DATA,allow_pickle=False); m=json.loads(MAN.read_text()); x=z['features'].astype(np.float32); off=z['offsets'].astype(np.int64); tar=z['targets'].astype(np.int64); cnt=z['candidate_counts'].astype(np.int64); meta=m['groups_meta'];
 ctx=[]
 for g in range(len(meta)):
  a,b=int(off[g]),int(off[g+1]); raw=x[a:b,15]; hq=float(np.clip((float(x[a:b,16].mean())+1.)/2.,0.,1.)) if b>a else 0.
  ctx.append(set_context(raw,int(cnt[g]),float(meta[g].get('source_length',0)),float(meta[g].get('source_variance',0)),hq))
 return x,off,tar,cnt,meta,np.asarray(ctx,np.float32),m
def full_input(xg,ctx): return np.concatenate([xg,np.repeat(ctx[None,:],len(xg),axis=0)],axis=1).astype(np.float32)
def group_metrics(model,mean,std,x,off,tar,cnt,meta,ctx,ids,device):
 raw_top1=rerank_top1=raw_top5=rerank_top5=0; match=defer=0; defer_pred=0; defer_correct=0; defer_tp=0; harm=rescue=0; teacher=[]; pred=[]; rank_loss=[]
 for g in ids:
  a,b=int(off[g]),int(off[g+1]); raw=x[a:b,15]; n=b-a; target=int(tar[g]);
  if not n: continue
  order=stable_raw_topk(raw,n); rscore,p,_=numpy_predict(model,x[a:b],ctx[g],raw,mean,std,device); rorder=stable_raw_topk(rscore,n)
  if target<n:
   match+=1; raw_top1+=int(order[0]==target); rerank_top1+=int(rorder[0]==target); raw_top5+=int(target in set(order[:5])); rerank_top5+=int(target in set(rorder[:5]));
   if order[0]==target and rorder[0]!=target: harm+=1
   if order[0]!=target and rorder[0]==target: rescue+=1
  else: defer+=1
  # The head predicts DEFER probability; bridge use is its complement.
  defer_flag=p>=.5; use=not defer_flag; defer_pred+=int(defer_flag); defer_correct+=int(defer_flag==(target>=n)); defer_tp+=int(defer_flag and target>=n);
  bridge=stable_raw_topk(x[a:b,16],min(2,n)); bmargin=float(x[a+bridge[0],16]-x[a+bridge[1],16]) if len(bridge)>1 else 0.; rmargin=float(raw[order[0]]-raw[order[1]]) if len(order)>1 else 0.; teacher_use=float(meta[g].get('support_quality',0))>=.2 and bmargin>=rmargin+.005; teacher.append(int(teacher_use)); pred.append(int(use));
 return {'groups':len(ids),'match_groups':match,'defer_groups':defer,'raw_top1_recall':raw_top1/max(1,match),'reranked_top1_recall':rerank_top1/max(1,match),'raw_top5_recall':raw_top5/max(1,match),'reranked_top5_recall':rerank_top5/max(1,match),'harm':harm,'rescue':rescue,'net_rescue':rescue-harm,'defer_precision':defer_tp/max(1,defer_pred),'defer_accuracy':defer_correct/max(1,len(pred)),'predicted_defer_groups':defer_pred,'bridge_use_rate':(len(pred)-defer_pred)/max(1,len(pred)),'teacher_use_rate':float(np.mean(teacher)) if teacher else 0.,'teacher_agreement':float(np.mean(np.asarray(teacher)==np.asarray(pred))) if teacher else 0.}
def train_fold(fold,tag,epochs,step_limit,device):
 x,off,tar,cnt,meta,ctx,m=load_data(); fd=m['folds'][str(fold)]; fit=[int(g) for g in fd['fit_groups']]; val=[int(g) for g in fd['validation_groups']]; match=[g for g in fit if tar[g] < off[g+1]-off[g]]; de=[g for g in fit if tar[g]>=off[g+1]-off[g]]
 if not match or not de: raise RuntimeError(f'fold {fold} lacks match/defer fit groups ({len(match)}/{len(de)})')
 comp=OUT/'completion';met=OUT/'metrics'; marker=comp/f'support_reranker_{tag}_f{fold}.launched';done=comp/f'support_reranker_{tag}_f{fold}.done'; cp=CK/f'support_reranker_{tag}_f{fold}_step{step_limit if step_limit else epochs:06d}.pt'
 if done.exists(): return json.loads((met/f'support_reranker_{tag}_f{fold}.json').read_text())
 if marker.exists(): raise RuntimeError(f'unit already launched without done: {marker}')
 atom_json(marker,{'phase':'Phase85','route':'B85S_RAW_ANCHORED_RERANK_DEFER','tag':tag,'fold':fold,'pid':os.getpid(),'gpu':str(device),'epochs':epochs,'step_limit':step_limit,'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
 fitrows=np.concatenate([np.arange(off[g],off[g+1]) for g in fit]); full=x[fitrows]; mean=full.mean(0).astype(np.float32); std=np.where(full.std(0)<1e-5,1.,full.std(0)).astype(np.float32)
 model=SupportReranker(19,10,64,.05).to(device); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-5); rng=np.random.default_rng(85000+fold); steps=0; losses=[]; max_bal=max(len(match),len(de));
 while True:
  balanced=np.concatenate([rng.choice(match,max_bal,replace=True),rng.choice(de,max_bal,replace=True)]);rng.shuffle(balanced)
  for g in balanced:
   a,b=int(off[g]),int(off[g+1]); X=torch.from_numpy((x[a:b]-mean)/std).to(device); c=torch.from_numpy(ctx[g]).to(device); raw=torch.from_numpy(x[a:b,15]).to(device); n=b-a; target=int(tar[g]); delta,dl=model(X,c); score=raw+delta; loss=F.binary_cross_entropy_with_logits(dl,torch.tensor(float(target>=n),device=device));
   if target<n:
    # Softmax over the registered raw-anchored candidate set; target labels
    # are TRAIN GT metadata and never part of the model input.
    loss=loss+F.cross_entropy(score.unsqueeze(0)/0.1,torch.tensor([target],device=device))
   opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);opt.step();steps+=1;losses.append(float(loss.detach().cpu()))
   if steps%1000==0:
    atom_torch(CK/f'support_reranker_{tag}_f{fold}_step{steps:06d}.pt',{'model':model.state_dict(),'mean':mean,'std':std,'step':steps,'fold':fold,'feature_dim':19,'candidate_dim':19,'context_dim':10,'residual_scale':.05,'manifest_sha256':sha(MAN),'config_sha256':sha(CFG)})
   if step_limit and steps>=step_limit: break
  if step_limit and steps>=step_limit:break
  if step_limit==0 and steps>=epochs*2*max_bal:break
  if step_limit==0 and steps>=max(1,epochs)*2*max_bal:break
 cp=CK/f'support_reranker_{tag}_f{fold}_step{steps:06d}.pt'; atom_torch(cp,{'model':model.state_dict(),'mean':mean,'std':std,'step':steps,'fold':fold,'feature_dim':19,'candidate_dim':19,'context_dim':10,'residual_scale':.05,'manifest_sha256':sha(MAN),'config_sha256':sha(CFG)})
 fm=group_metrics(model,mean,std,x,off,tar,cnt,meta,ctx,fit,device); vm=group_metrics(model,mean,std,x,off,tar,cnt,meta,ctx,val,device); obj={'schema_version':'trackocd.phase85.support_reranker_metrics.v1','phase':'Phase85 B85S','route':'B85S_RAW_ANCHORED_SET_RERANK_DEFER','tag':tag,'fold':fold,'epochs':epochs,'steps':steps,'feature_dim':19,'candidate_dim':19,'context_dim':10,'fit_groups':len(fit),'validation_groups':len(val),'fit_metrics':fm,'validation_metrics':vm,'loss_first':losses[0] if losses else None,'loss_last':losses[-1] if losses else None,'checkpoint':str(cp.resolve()),'checkpoint_sha256':sha(cp),'manifest':str(MAN.resolve()),'manifest_sha256':sha(MAN),'config':str(CFG.resolve()),'config_sha256':sha(CFG),'balanced_sampling':{'fit_match':len(match),'fit_defer':len(de),'replacement':True},'public_dev_q1_sealed_accessed':False,'future_rows_or_tracks':False,'ids_as_model_input':False,'gt_fields_in_feature_tensor':False,'device':device}
 obj['device']=str(device); atom_json(met/f'support_reranker_{tag}_f{fold}.json',obj);atom_json(done,{'status':'DONE','fold':fold,'tag':tag,'checkpoint':str(cp.resolve()),'metrics':str((met/f'support_reranker_{tag}_f{fold}.json').resolve())});return obj
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int,required=True);ap.add_argument('--tag',required=True);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--steps',type=int,default=0);ap.add_argument('--device',default='cpu');a=ap.parse_args(); torch.set_num_threads(4); device=torch.device(a.device if a.device!='cpu' and torch.cuda.is_available() else 'cpu'); print(json.dumps(train_fold(a.fold,a.tag,a.epochs,a.steps,device),indent=2,sort_keys=True))
if __name__=='__main__':main()
