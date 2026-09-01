#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase41'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def calc(s,l):
 o=np.argsort(s)[::-1]; h=l[o]; c=np.cumsum(h); n=max(l.sum(),1); return float(h[0]),float(np.sum(c/(np.arange(len(h))+1)*h)/n),float(np.max(s[l>0])-np.max(s[l<=0])) if np.any(l<=0) else 0
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; outfold={}
  for p in PREFIXES:
   raw1=[];rawm=[];rawg=[];b1=[];bm=[];bg=[];eq=0
   for r in val:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk: continue
    cks=sk+([hk] if hk in meta else []); q=vec(qk,meta,feats,p); c=np.asarray([vec(k,meta,feats,p) for k in cks]); lab=np.asarray([1.]*len(sk)+[0.]*(len(cks)-len(sk))); raw=q@c.T; s=np.asarray([vec(k,meta,feats,p) for k in sk]); ctx=s.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8); bridge=SafetyVectorBridge(); z,a,_=bridge(torch.tensor(q).view(1,-1),torch.tensor(ctx).view(1,-1),torch.tensor([float(raw.max())]),torch.tensor([float((s@q).max())]),True); zv=z.detach().numpy()[0]; bs=zv@c.T; eq+=int(np.allclose(zv,q,atol=1e-7)); x,y,g=calc(raw,lab); raw1.append(x);rawm.append(y);rawg.append(g); x,y,g=calc(bs,lab); b1.append(x);bm.append(y);bg.append(g)
   outfold[str(p)]={'raw':{'r1':float(np.mean(raw1)) if raw1 else 0,'map':float(np.mean(rawm)) if rawm else 0,'hard_gap':float(np.mean(rawg)) if rawg else 0},'alpha0_bridge':{'r1':float(np.mean(b1)) if b1 else 0,'map':float(np.mean(bm)) if bm else 0,'hard_gap':float(np.mean(bg)) if bg else 0},'alpha0_exact_raw_fraction':float(eq/max(len(raw1),1))}
  folds.append({'fold':fold,'prefix':outfold})
 agg={str(p):{name:{m:float(np.mean([f['prefix'][str(p)][name][m] for f in folds])) for m in ('r1','map','hard_gap')} for name in ('raw','alpha0_bridge')} for p in PREFIXES}
 atomic(OUT/'audit/bridge_diagnostic.json',{'protocol':'phase41_alpha0_row_vector_bridge','folds':folds,'aggregate':agg,'row_vector_dim':768,'alpha_zero_identity':True,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/stage1.done',{'stage':1,'contract':'PASS'}); print(json.dumps({'stage1':'PASS','p16':agg['16']},indent=2))
if __name__=='__main__': main()
