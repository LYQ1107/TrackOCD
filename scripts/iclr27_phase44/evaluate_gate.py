#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase44.selective import CalibratedGate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase44'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def calc(s,l):
 o=np.argsort(s)[::-1]; h=l[o]; c=np.cumsum(h); n=max(l.sum(),1); return float(h[0]),float(np.sum(c/(np.arange(len(h))+1)*h)/n),float(np.max(s[l>0])-np.max(s[l<=0])) if np.any(l<=0) else 0
def main():
 rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; gate=CalibratedGate(); gate.load_state_dict(torch.load(OUT/f'checkpoints/calibrated_formal_fix1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); gate.eval(); bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval(); per={}
  for p in PREFIXES:
   rr=[];rm=[];rg=[];sr=[];sm=[];sg=[];use=[];agree=[];unsafe=[]
   for r in val:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk or hk not in meta: continue
    cks=sk+[hk]; q=vec(qk,meta,feats,p); c=np.asarray([vec(k,meta,feats,p) for k in cks]); lab=np.asarray([1.]*len(sk)+[0.]); raw=q@c.T; sv=np.asarray([vec(k,meta,feats,p) for k in sk]); ctx=sv.mean(0);ctx/=max(float(np.linalg.norm(ctx)),1e-8); sq=float((sv@q).max())
    with torch.no_grad(): z,a,_=bridge(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([sq]),True); bs=z.numpy()[0]@c.T; rmarg=torch.tensor([float(raw[:len(sk)].max()-raw[len(sk):].max())]); bmarg=torch.tensor([float(bs[:len(sk)].max()-bs[len(sk):].max())]); pr=gate(rmarg,bmarg,torch.tensor([sq]),a,torch.zeros(1)).item()
    chosen=bs if pr>=0.5 else raw; x,y,g=calc(raw,lab);rr.append(x);rm.append(y);rg.append(g);x,y,g=calc(chosen,lab);sr.append(x);sm.append(y);sg.append(g);use.append(pr>=0.5);unsafe.append(bool(pr>=.5 and raw.argmax()<len(sk) and bs.argmax()>=len(sk)));teacher=bool(sq>=.2 and float(bmarg)>=float(rmarg)+.005);agree.append((pr>=.5)==teacher)
   per[str(p)]={'raw':{'r1':float(np.mean(rr)) if rr else 0,'map':float(np.mean(rm)) if rm else 0,'hard_gap':float(np.mean(rg)) if rg else 0},'learned':{'r1':float(np.mean(sr)) if sr else 0,'map':float(np.mean(sm)) if sm else 0,'hard_gap':float(np.mean(sg)) if sg else 0},'bridge_use_rate':float(np.mean(use)) if use else 0,'teacher_agreement':float(np.mean(agree)) if agree else 0,'unsafe_flip_rate':float(np.mean(unsafe)) if unsafe else 0}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{n:{m:float(np.mean([f['prefix'][str(p)][n][m] for f in folds])) for m in ('r1','map','hard_gap')} for n in ('raw','learned')} for p in PREFIXES}; atomic(OUT/'metrics/calibrated_retrieval.json',{'protocol':'phase44_calibrated_conditional_gate','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/retrieval.done',{'metrics':str(OUT/'metrics/calibrated_retrieval.json'),'p16':agg['16']}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__':main()
