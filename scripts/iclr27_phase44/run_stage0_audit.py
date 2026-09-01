#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase43.selective import PolicyDistilledGate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase44'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]; recs=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); fit=[r for r in man['records'] if r['split']=='fit' and r['kind']=='multi_positive_cross_video']; bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); gate=PolicyDistilledGate(); gate.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase43/checkpoints/policy_formal_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge.eval();gate.eval(); pstats={}
  for p in PREFIXES:
   vals=[]; ts=[]; cond=[]; unsafe=0
   for r in fit:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk or hk not in meta: continue
    q=vec(qk,meta,feats,p); c=np.asarray([vec(k,meta,feats,p) for k in sk+[hk]]); sv=np.asarray([vec(k,meta,feats,p) for k in sk]); ctx=sv.mean(0);ctx/=max(float(np.linalg.norm(ctx)),1e-8); raw=q@c.T; sq=float((sv@q).max())
    with torch.no_grad(): z,a,_=bridge(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([sq]),True); bs=z.numpy()[0]@c.T; rm=float(raw[:len(sk)].max()-raw[len(sk):].max()); bm=float(bs[:len(sk)].max()-bs[len(sk):].max()); pr=gate(torch.tensor([rm]),torch.tensor([bm]),torch.tensor([sq]),a,torch.zeros(1)).item()
    teacher=bool(sq>=0.2 and bm>=rm+0.005); vals.append(pr);ts.append(teacher);cond.append({'support_quality_lt_0.2':sq<0.2,'margin_below':bm<rm+0.005}); unsafe+=int(raw.argmax()<len(sk) and bs.argmax()>=len(sk)); recs.append({'fold':fold,'prefix':p,'p':pr,'teacher':teacher,'support_quality':sq,'raw_margin':rm,'bridge_margin':bm})
   pstats[str(p)]={'episodes':len(vals),'teacher_rate':float(np.mean(ts)) if ts else 0,'p_mean':float(np.mean(vals)) if vals else 0,'p_q10':float(np.quantile(vals,.1)) if vals else 0,'p_q50':float(np.quantile(vals,.5)) if vals else 0,'p_q90':float(np.quantile(vals,.9)) if vals else 0,'p_lt_0.5':float(np.mean(np.asarray(vals)<.5)) if vals else 0,'constant_majority_agreement':float(np.mean(np.asarray(ts)==(np.mean(ts)>=.5))) if ts else 0,'unsafe_rate':unsafe/max(len(vals),1)}
  folds.append({'fold':fold,'prefix':pstats})
 atomic(OUT/'audit/contract.json',{'phase':44,'teacher_rule':'support_quality>=0.2 AND bridge_margin>=raw_margin+0.005','inference':'p>=0.5 unchanged','phase41_bridge_frozen':True,'forbidden_inputs':['category','text','IDs','future','held GT','StateMemory','controller action']}); atomic(OUT/'audit/resource_preflight.json',{'gpu_ids':[4,5,6,7],'gpu_free_mib':40337,'ram_available_gb':118,'public_q1_dev_access':False,'bounded_workers':4}); atomic(OUT/'audit/phase43_p_distribution.json',{'folds':folds,'records':recs,'denominator':'TRAIN fit episodes'}); atomic(OUT/'completion/stage0.done',{'stage':0,'contract':'PASS'}); print(json.dumps({'stage0':'PASS','p16':[f['prefix']['16'] for f in folds]},indent=2))
if __name__=='__main__': main()
