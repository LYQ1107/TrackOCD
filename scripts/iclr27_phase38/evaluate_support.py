#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase38'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats):
 x=feats[np.asarray(meta[k]['rows'])[:16]]; z=x.mean(0); return z/max(np.linalg.norm(z),1e-8)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT/('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json'%fold))); val=[r for r in man['records'] if r['split']=='val']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); ck=torch.load(OUT/f'checkpoints/support_formal_f{fold}_best.pt',map_location='cpu',weights_only=False); model=SupportSetCorrespondenceEncoder(); model.load_state_dict(ck['model']); model.eval(); r1=[];r5=[];aps=[]
  allv=np.asarray([vec(k,meta,feats) for k in keys]); vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys])
  for i,k in enumerate(keys):
   ci=np.arange(len(keys)); ci=ci[(ci!=i)&(vids!=vids[i])]; pos=ci[cats[ci]==cats[i]]
   if len(pos)==0: continue
   with torch.no_grad():
    q=model.encode_track(torch.tensor(allv[i]).view(1,1,-1),torch.ones(1,1,dtype=torch.bool)).numpy()[0]
    cv=model.encode_track(torch.tensor(allv[ci]).unsqueeze(1),torch.ones(len(ci),1,dtype=torch.bool)).numpy(); score=q@cv.T; order=ci[np.argsort(score)[::-1]]
   hit=np.array([j in set(pos.tolist()) for j in order],float); r1.append(float(hit[:1].max(initial=0)));r5.append(float(hit[:5].max(initial=0)));cum=np.cumsum(hit);aps.append(float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(len(pos),1)))
  folds.append({'fold':fold,'tracklets':len(keys),'r1':float(np.mean(r1)) if r1 else 0,'r5':float(np.mean(r5)) if r5 else 0,'map':float(np.mean(aps)) if aps else 0})
 agg={k:float(np.mean([f[k] for f in folds])) for k in ('r1','r5','map')}; atomic(OUT/'metrics/support_retrieval.json',{'protocol':'phase38_prior_completed_support_retrieval','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/GT']}); atomic(OUT/'completion/retrieval.done',{'aggregate':agg}); print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
