#!/usr/bin/env python3
"""Evaluate frozen Phase30 support-set encoders on TRAIN-disjoint retrieval only."""
import json, tempfile, os
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata, retrieval
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase30'; PREFIXES=(1,2,4,8,16)
def atomic(path,val):
 path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(dir=str(path.parent),prefix='.'+path.name); 
 with os.fdopen(fd,'w') as f: json.dump(val,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,path)
def embed(model,keys,meta,feats,prefix,device):
 arr=np.zeros((len(keys),16,feats.shape[1]),np.float32); mask=np.zeros((len(keys),16),bool)
 for i,k in enumerate(keys):
  inds=meta[k]['rows'][:min(prefix,16)]; arr[i,:len(inds)]=feats[np.asarray(inds)]; mask[i,:len(inds)]=1
 with torch.no_grad():
  out=model.encode_track(torch.from_numpy(arr).to(device),torch.from_numpy(mask).to(device)).cpu().numpy()
 return out
def main():
 torch.set_num_threads(1); rows,tracks,feats=load_tracks(); meta=track_metadata(rows,tracks); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); folds=[]
 for fold in range(4):
  man=json.loads((OUT/f'manifests/episode_manifest_f{fold}.json').read_text()); val=[r for r in man['records'] if r['split']=='val']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); support={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in keys] for r in val if r['kind']=='multi_positive_cross_video' and r['query_track_key'] in meta}; support={k:v for k,v in support.items() if v}
  ckpath=OUT/f'checkpoints/interface_formal_f{fold}_best.pt'; model=SupportSetCorrespondenceEncoder(); ck=torch.load(ckpath,map_location='cpu',weights_only=False); model.load_state_dict(ck['model']); model.to(device).eval(); per={}
  for p in PREFIXES:
   e=embed(model,keys,meta,feats,p,device); per[str(p)]={'raw':retrieval(keys,e,meta),'support_set':retrieval(keys,e,meta,support)}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':per,'checkpoint':str(ckpath),'best_score':ck.get('best_score')})
 agg={}
 for p in PREFIXES:
  agg[str(p)]={}
  for mode in ('raw','support_set'):
   vals=[f['prefix'][str(p)][mode] for f in folds]; agg[str(p)][mode]={m:float(np.mean([v[m] for v in vals])) for m in ('r1','r5','map','hard_negative_gap','positive_coverage')}
 result={'protocol':'trackocd_iclr27_phase30_stage2_frozen_interface_retrieval','folds':folds,'aggregate':agg,'checkpoint_selection':'fold-internal validation only','sealed_inputs_not_read':['DEV+','Q1','public new-model labels','held event outcomes','future rows/tracks','IDs/text/GT in model input']}
 atomic(OUT/'metrics/interface_retrieval.json',result); atomic(OUT/'audit/interface_summary.json',{'aggregate':agg,'folds':[{'fold':f['fold'],'tracklets':f['validation_tracklets']} for f in folds]}); atomic(OUT/'completion/stage2.done',{'folds':4,'protocol':result['protocol']})
 print(json.dumps({'stage2':'done','metrics':str(OUT/'metrics/interface_retrieval.json'),'aggregate':agg},indent=2))
if __name__=='__main__': main()
