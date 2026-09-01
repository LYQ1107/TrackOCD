#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np,torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks,track_metadata
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase39'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats,p):
 x=feats[np.asarray(meta[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(np.linalg.norm(z),1e-8)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); folds=[]
 for fold in range(4):
  man=json.load(open(ROOT / ('outputs/iclr27_phase30/manifests/episode_manifest_f%d.json' % fold))); val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']; ck=torch.load(ROOT/f'outputs/iclr27_phase38/checkpoints/support_formal_f{fold}_best.pt',map_location='cpu',weights_only=False); model=SupportSetCorrespondenceEncoder(); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   old_r1=[];new_r1=[];old_ap=[];new_ap=[];gaps=[]
   for r in val:
    qk=r['query_track_key']; sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key');
    if qk not in meta or not sk: continue
    cands=sk+([hk] if hk in meta else []); qv=vec(qk,meta,feats,p); cv=np.asarray([vec(k,meta,feats,p) for k in cands]); labels=np.array([1]*len(sk)+[0]*(len(cands)-len(sk)))
    with torch.no_grad():
     q=torch.tensor(qv).view(1,1,-1); s=torch.tensor(cv).view(1,len(cands),1,-1); qm=torch.ones(1,1,dtype=torch.bool); sm=torch.ones(1,len(cands),1,dtype=torch.bool); setm=torch.ones(1,len(cands),dtype=torch.bool); out=model(q,qm,s,sm,setm); ns=out['pair_scores'][0].numpy(); qe=model.encode_track(q,qm).numpy()[0]; ce=model.encode_track(torch.tensor(cv).unsqueeze(1),torch.ones(len(cands),1,dtype=torch.bool)).numpy(); old=qe@ce.T
    for scores,r1s,aps in ((old,old_r1,old_ap),(ns,new_r1,new_ap)):
     order=np.argsort(scores)[::-1]; hit=labels[order].astype(float); r1s.append(float(hit[0])); cum=np.cumsum(hit); aps.append(float(np.sum(cum/(np.arange(len(hit))+1)*hit)/max(labels.sum(),1)))
    if len(cands)>len(sk): gaps.append(float(ns[:len(sk)].max()-ns[len(sk):].max()))
   per[str(p)]={'old_incorrect':{'r1':float(np.mean(old_r1)) if old_r1 else 0.0,'map':float(np.mean(old_ap)) if old_ap else 0.0},'correct_forward':{'r1':float(np.mean(new_r1)) if new_r1 else 0.0,'map':float(np.mean(new_ap)) if new_ap else 0.0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0.0},'episodes':len(new_r1)}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{m:float(np.mean([f['prefix'][str(p)]['correct_forward'][m] for f in folds])) for m in ('r1','map','hard_negative_gap')} for p in PREFIXES}; atomic(OUT/'metrics/correct_replay.json',{'protocol':'phase39_support_conditioned_forward_replay','folds':folds,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/stage1.done',{'stage':1,'metrics':str(OUT/'metrics/correct_replay.json')}); print(json.dumps({'p16':agg['16']},indent=2))
if __name__=='__main__': main()
