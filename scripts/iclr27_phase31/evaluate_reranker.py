#!/usr/bin/env python3
import json,os,tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata,retrieval
from scripts.iclr27_phase31.train_reranker import pair
from src.iclr27_phase31.reranker import MonotonicRawReranker
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase31'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 torch.set_num_threads(1); rows,tracks,feats=load_tracks(); meta=track_metadata(rows,tracks); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); keys=sorted({r['query_track_key'] for r in man['records'] if r['split']=='val' and r['query_track_key'] in meta}); ck=torch.load(OUT/f'checkpoints/rawrerank_formal_f{fold}_best.pt',map_location='cpu',weights_only=False); model=MonotonicRawReranker(); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   vec=[]
   for k in keys:
    inds=meta[k]['rows'][:min(p,16)]; v=feats[np.asarray(inds)].mean(0); v/=max(np.linalg.norm(v),1e-8); vec.append(v)
   vec=np.asarray(vec,np.float32); vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys]); allidx=np.arange(len(keys)); r1=[];r5=[];aps=[];gaps=[]
   for i,q in enumerate(keys):
    cand=allidx[(allidx!=i)&(vids!=vids[i])]; pos=cand[cats[cand]==cats[i]]; neg=cand[cats[cand]!=cats[i]]
    if len(pos)==0 or len(neg)==0: continue
    raws=[]; ms=[]
    for j in cand:
      ra,fa=pair(q,keys[j],meta,feats); raws.append(ra); ms.append(fa)
    with torch.no_grad(): score=model(torch.tensor(raws),torch.tensor(np.asarray(ms))).numpy()
    order=cand[np.argsort(score)[::-1]]; hit=np.array([j in set(pos.tolist()) for j in order],float); r1.append(float(hit[:1].max(initial=0)));r5.append(float(hit[:5].max(initial=0)));cum=np.cumsum(hit);aps.append(float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(len(pos),1)));gaps.append(float(np.max(score[np.isin(cand,pos)])-np.max(score[np.isin(cand,neg)])))
   per[str(p)]={'queries':len(r1),'r1':float(np.mean(r1)) if r1 else 0.0,'r5':float(np.mean(r5)) if r5 else 0.0,'map':float(np.mean(aps)) if aps else 0.0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0.0,'positive_coverage':float(len(r1)/max(len(keys),1))}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':per})
 agg={str(p):{m:float(np.mean([f['prefix'][str(p)][m] for f in folds])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')} for p in PREFIXES}
 base=json.load(open(ROOT/'outputs/iclr27_phase31/audit/stage1_summary.json'))['raw_cosine_p16']; deltas={'r1':agg['16']['r1']-base['r1'],'map':agg['16']['map']-base['map'],'hard_negative_gap':agg['16']['hard_negative_gap']-base['hard_negative_gap']}
 result={'protocol':'trackocd_iclr27_phase31_raw_space_reranker_validation','folds':folds,'aggregate':agg,'baseline_p16':base,'delta_p16':deltas,'gate_r31':{'pass':False,'reason':'computed after frozen validation; controller compatibility requires preregistered thresholds and is not run in this diagnostic'},'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','held outcomes','future','IDs/text/GT model input']}; atomic(OUT/'metrics/reranker_validation.json',result); atomic(OUT/'completion/stage3_validation.done',{'gate_r31':'FAIL','metrics':str(OUT/'metrics/reranker_validation.json')}); print(json.dumps({'aggregate_p16':agg['16'],'delta':deltas},indent=2))
if __name__=='__main__': main()
