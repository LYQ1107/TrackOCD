#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase40.residual import RawPreservingSupportResidual
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase40'; TAG=os.environ.get('PHASE40_TAG','residual_formal'); PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 x=f[np.asarray(m[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def calc(sc,lab):
 o=np.argsort(sc)[::-1]; h=lab[o]; c=np.cumsum(h); n=max(lab.sum(),1); return float(h[0]),float(h[:5].max(initial=0)),float(np.sum(c/(np.arange(len(h))+1)*h)/n)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json')); val=[r for r in man['records'] if r['split']=='val']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); smap={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val if r['kind']=='multi_positive_cross_video'}; model=RawPreservingSupportResidual().to(dev); ck=torch.load(OUT/f'checkpoints/{TAG}_f{fold}_best.pt',map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.eval(); per={}
  vids=np.asarray([meta[k]['video'] for k in keys]); cats=np.asarray([meta[k]['category'] for k in keys]);
  for p in PREFIXES:
   mat=np.asarray([vec(k,meta,feats,p) for k in keys],np.float32); n=len(keys); raw_scores=[None]*n; res_scores=[None]*n
   with torch.no_grad():
    tm=torch.tensor(mat,device=dev); vm=torch.tensor(vids,device=dev)
    for st in range(0,n,128):
      en=min(n,st+128); q=torch.tensor(mat[st:en],device=dev); rawm=q@tm.T; outm=rawm.clone();
      for bi,i in enumerate(range(st,en)):
       cand=np.where((vids!=vids[i]) & (np.arange(n)!=i))[0]; qv=mat[i]; cv=mat[cand]; sk=smap.get(keys[i],[]); sv=np.asarray([mat[keys.index(k)] for k in sk],np.float32) if sk else np.empty((0,mat.shape[1]),np.float32); raw=cv@qv; sup=(sv@cv.T).max(0) if len(sk) else np.zeros(len(cand),np.float32); qt=torch.tensor(qv,device=dev).view(1,-1).expand(len(cand),-1); ct=torch.tensor(cv,device=dev); rt=torch.tensor(raw,device=dev); smax=torch.tensor(sup,device=dev); score,_,_=model(rt,qt,ct,smax,torch.full_like(rt,float(len(sk))),valid_support=bool(sk)); raw_scores[i]=(cand,raw); res_scores[i]=(cand,score.cpu().numpy())
   rr=[];rm=[];r5=[];sr=[];sm=[];rg=[];sg=[]
   for i,k in enumerate(keys):
    cand,raw=raw_scores[i]; lab=(cats[cand]==cats[i]).astype(np.float32)
    if not lab.any(): continue
    _,b,c=calc(raw,lab); a,_,_=calc(raw,lab); rr.append(a);r5.append(b);rm.append(c); _,rs5,rsc=calc(res_scores[i][1],lab); sa,_,_=calc(res_scores[i][1],lab); sr.append(sa);sm.append(rsc); rg.append(float(np.max(raw[lab>0])-np.max(raw[lab<=0])) if np.any(lab<=0) else 0); sg.append(float(np.max(res_scores[i][1][lab>0])-np.max(res_scores[i][1][lab<=0])) if np.any(lab<=0) else 0)
   per[str(p)]={'queries':len(rr),'raw':{'r1':float(np.mean(rr)) if rr else 0,'r5':float(np.mean(r5)) if r5 else 0,'map':float(np.mean(rm)) if rm else 0,'hard_negative_gap':float(np.mean(rg)) if rg else 0},'residual':{'r1':float(np.mean(sr)) if sr else 0,'r5':float(np.mean(r5)) if r5 else 0,'map':float(np.mean(sm)) if sm else 0,'hard_negative_gap':float(np.mean(sg)) if sg else 0}}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':per})
 agg={str(p):{name:{m:float(np.mean([f['prefix'][str(p)][name][m] for f in folds])) for m in ('r1','r5','map','hard_negative_gap')} for name in ('raw','residual')} for p in PREFIXES}
 atomic(OUT/'metrics/global_retrieval.json',{'protocol':'phase40_global_cross_video_raw_preserving','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future rows/tracks','IDs/text/held GT']}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
