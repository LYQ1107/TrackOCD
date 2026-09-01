#!/usr/bin/env python3
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase46.selective import ConditionalLogitGate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase46'; PREFIXES=(1,2,4,8,16)
def atomic(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='.'+path.name)
 with os.fdopen(fd,'w') as f: json.dump(value,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def metrics(scores, labels):
 o=np.argsort(scores)[::-1]; h=labels[o]; c=np.cumsum(h); n=max(labels.sum(),1); ap=float(np.sum(c/(np.arange(len(h))+1)*h)/n); r1=float(h[0])
 pos=scores[labels>0]; neg=scores[labels<=0]; gap=float(pos.max()-neg.max()) if len(pos) and len(neg) else 0.0
 return r1,ap,gap
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']
  gate=ConditionalLogitGate(); gate.load_state_dict(torch.load(OUT/f'checkpoints/phase46_formal_v1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); gate.eval(); bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); per={}
  for p in PREFIXES:
   rawm=[]; learnm=[]; uses=[]; agrees=[]; uns=[]
   for r in val:
    qk=r.get('query_track_key'); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key')
    if qk not in meta or not sk or hk not in meta: continue
    q=vec(qk,meta,feats,p); sv=np.asarray([vec(k,meta,feats,p) for k in sk]); h=vec(hk,meta,feats,p); c=np.concatenate([sv,[h]],0); lab=np.asarray([1.]*len(sk)+[0.]); raw=q@c.T; ctx=sv.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8)
    with torch.no_grad():
      qt=torch.tensor(q).view(1,-1); ct=torch.tensor(c); sq=torch.tensor([(sv@q).max()]); z,a,_=bridge(qt,torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),sq,True); bs=(z@ct.T).numpy()[0]; rm=float(raw[:len(sk)].max()-raw[len(sk):].max()); bm=float(bs[:len(sk)].max()-bs[len(sk):].max()); logit=gate(torch.tensor([rm]),torch.tensor([bm]),sq,a,torch.zeros(1)); use=bool(logit.item()>=0.0)
    rawm.append(metrics(raw,lab)); learnm.append(metrics(bs if use else raw,lab)); uses.append(use); uns.append(bool(use and raw.argmax()<len(sk) and bs.argmax()>=len(sk))); teacher=bool(float(sq.item())>=.2 and bm>=rm+.005); agrees.append(use==teacher)
   def avg(i): return float(np.mean([x[i] for x in rawm])) if rawm else 0.0
   def avg_l(i): return float(np.mean([x[i] for x in learnm])) if learnm else 0.0
   per[str(p)]={'raw':{'r1':avg(0),'map':avg(1),'hard_gap':avg(2)},'learned':{'r1':avg_l(0),'map':avg_l(1),'hard_gap':avg_l(2)},'bridge_use_rate':float(np.mean(uses)) if uses else 0.0,'teacher_agreement':float(np.mean(agrees)) if agrees else 0.0,'unsafe_flip_rate':float(np.mean(uns)) if uns else 0.0,'queries':len(rawm)}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{name:{metric:float(np.mean([f['prefix'][str(p)][name][metric] for f in folds])) for metric in ('r1','map','hard_gap')} for name in ('raw','learned')} for p in PREFIXES}
 atomic(OUT/'metrics/phase46_retrieval.json',{'protocol':'phase46_full_materialized_balanced_bce_conditioned_gate','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/retrieval.done',{'metrics':str(OUT/'metrics/phase46_retrieval.json'),'p16':agg['16']}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
