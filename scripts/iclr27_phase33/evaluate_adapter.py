#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase33.adapter import QueryConditionedAdapter
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase33'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def v(k,meta,feats,p):
 x=feats[np.asarray(meta[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(np.linalg.norm(z),1e-8)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); val=[r for r in man['records'] if r['split']=='val']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); supp={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val if r['kind']=='multi_positive_cross_video'}; ck=torch.load(OUT/f'checkpoints/adapter_formal_f{fold}_best.pt',map_location='cpu',weights_only=False); model=QueryConditionedAdapter(); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   cand=np.asarray([v(k,meta,feats,p) for k in keys]); vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys]); r1=[];r5=[];aps=[];gaps=[]
   for i,k in enumerate(keys):
    ci=np.arange(len(keys)); ci=ci[(ci!=i)&(vids!=vids[i])]; pos=ci[cats[ci]==cats[i]]; neg=ci[cats[ci]!=cats[i]]
    if len(pos)==0 or len(neg)==0: continue
    sk=supp.get(k,[]); ss=np.asarray([v(s,meta,feats,p) for s in sk if s in meta]); raw=torch.tensor(cand[i]).unsqueeze(0); st=torch.tensor(ss).unsqueeze(0) if len(ss) else None
    with torch.no_grad(): q=model(raw,st).numpy()[0]
    score=q@cand[ci].T; order=ci[np.argsort(score)[::-1]]; ps=set(pos.tolist()); hit=np.array([j in ps for j in order],float); r1.append(float(hit[:1].max(initial=0)));r5.append(float(hit[:5].max(initial=0)));cum=np.cumsum(hit);aps.append(float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(len(pos),1)));gaps.append(float(np.max(score[np.isin(ci,pos)])-np.max(score[np.isin(ci,neg)])))
   per[str(p)]={'queries':len(r1),'r1':float(np.mean(r1)) if r1 else 0,'r5':float(np.mean(r5)) if r5 else 0,'map':float(np.mean(aps)) if aps else 0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0,'positive_coverage':float(len(r1)/max(len(keys),1))}
  folds.append({'fold':fold,'tracklets':len(keys),'prefix':per})
 agg={str(p):{m:float(np.mean([f['prefix'][str(p)][m] for f in folds])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')} for p in PREFIXES}; result={'protocol':'trackocd_iclr27_phase33_adapter_retrieval','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/GT input']}; atomic(OUT/'metrics/adapter_retrieval.json',result); atomic(OUT/'completion/stage3_validation.done',{'stage':3,'metrics':str(OUT/'metrics/adapter_retrieval.json')}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
