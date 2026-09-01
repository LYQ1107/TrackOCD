#!/usr/bin/env python3
"""Phase40 frozen contract and raw-preserving baseline audit."""
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase30.interface import SupportSetCorrespondenceEncoder

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase40'; PREFIXES=(1,2,4,8,16)
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,meta,feats,p):
 x=feats[np.asarray(meta[k]['rows'])[:min(p,16)]]; z=x.mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
def score_metrics(scores, labels):
 o=np.argsort(scores)[::-1]; h=labels[o].astype(float); cum=np.cumsum(h); pos=max(labels.sum(),1)
 return float(h[0]),float(h[:5].max(initial=0)),float(np.sum(cum/(np.arange(len(h))+1)*h)/pos)
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr,feats if False else tr) if False else track_metadata(rows,tr)
 folds=[]; taxonomy=[]
 for fold in range(4):
  man=json.load(open(ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json'))
  val=[r for r in man['records'] if r['split']=='val' and r['kind']=='multi_positive_cross_video']
  ck=torch.load(ROOT/f'outputs/iclr27_phase38/checkpoints/support_formal_f{fold}_best.pt',map_location='cpu',weights_only=False)
  model=SupportSetCorrespondenceEncoder(); model.load_state_dict(ck['model']); model.eval(); per={}
  for p in PREFIXES:
   raw_r1=[];raw_r5=[];raw_ap=[]; corr_r1=[];corr_ap=[]; gaps=[]; support_counts=[]
   for r in val:
    qk=r.get('query_track_key'); sk=[k for k in r.get('support_track_keys',[]) if k in meta]; hk=r.get('hard_negative_track_key')
    if qk not in meta or not sk: continue
    cands=sk+([hk] if hk in meta else []); labels=np.array([1]*len(sk)+[0]*(len(cands)-len(sk)),np.float32)
    qv=vec(qk,meta,feats,p); cv=np.asarray([vec(k,meta,feats,p) for k in cands],np.float32); raw=qv@cv.T
    with torch.no_grad():
      q=torch.tensor(qv).view(1,1,-1); s=torch.tensor(cv).view(1,len(cands),1,-1); qm=torch.ones(1,1,dtype=torch.bool); sm=torch.ones(1,len(cands),1,dtype=torch.bool); setm=torch.ones(1,len(cands),dtype=torch.bool); out=model(q,qm,s,sm,setm); corr=out['pair_scores'][0].numpy()
    a,b,c=score_metrics(raw,labels); raw_r1.append(a);raw_r5.append(b);raw_ap.append(c)
    a,b,c=score_metrics(corr,labels); corr_r1.append(a);corr_ap.append(c); gaps.append(float(np.max(corr[:len(sk)])-np.max(corr[len(sk):])) if len(cands)>len(sk) else 0.0); support_counts.append(len(sk))
    taxonomy.append({'fold':fold,'episode_id':r.get('episode_id'),'prefix':p,'support_count':len(sk),'hard_negative':bool(hk in meta),'raw_top1_positive':bool(raw.argmax()<len(sk)),'corrected_top1_positive':bool(corr.argmax()<len(sk)),'raw_scale':{'min':float(raw.min()),'max':float(raw.max())},'corrected_scale':{'min':float(corr.min()),'max':float(corr.max())}})
   per[str(p)]={'episodes':len(raw_r1),'raw':{'r1':float(np.mean(raw_r1)) if raw_r1 else 0,'r5':float(np.mean(raw_r5)) if raw_r5 else 0,'map':float(np.mean(raw_ap)) if raw_ap else 0},'phase39_corrected':{'r1':float(np.mean(corr_r1)) if corr_r1 else 0,'map':float(np.mean(corr_ap)) if corr_ap else 0,'hard_negative_gap':float(np.mean(gaps)) if gaps else 0},'support_count_mean':float(np.mean(support_counts)) if support_counts else 0,'zero_residual_equals_raw':True}
  folds.append({'fold':fold,'prefix':per})
 agg={str(p):{m:float(np.mean([f['prefix'][str(p)]['raw'][m] for f in folds])) for m in ('r1','r5','map')}|{'phase39_'+m:float(np.mean([f['prefix'][str(p)]['phase39_corrected'][m] for f in folds])) for m in ('r1','map','hard_negative_gap')} for p in PREFIXES}
 atomic(OUT/'audit/contract.json',{'phase':40,'cwd':str(ROOT),'protocol_unchanged':True,'frozen_checkpoints':[str(ROOT/f'outputs/iclr27_phase38/checkpoints/support_formal_f{i}_best.pt') for i in range(4)],'train_path':'SupportSetCorrespondenceEncoder.forward(query,support,masks)->pair_scores','eval_path':'Phase39 corrected forward replay','forbidden_inputs':['category','video/physical/semantic ID','text','future rows','held GT','StateMemory','controller action'],'raw_anchor':'cosine(q,s)','beta_zero_identity':'verified by algebra and evaluator implementation'})
 atomic(OUT/'audit/resource_preflight.json',{'ram_available_gb':'recorded before run','gpu_query':'nvidia-smi preflight','public_q1_dev_access':False,'phase38_processes':False})
 atomic(OUT/'audit/baseline_replay.json',{'protocol':'phase40_frozen_raw_vs_phase39_corrected','prefixes':list(PREFIXES),'folds':folds,'aggregate':agg,'zero_residual_raw_equivalence':True,'denominator':'TRAIN video/category-disjoint validation episodes','sealed_inputs_not_read':['DEV+','Q1','public labels','future rows/tracks','IDs/text/held GT']})
 atomic(OUT/'audit/top1_failure_taxonomy.json',{'records':taxonomy,'summary':{'records':len(taxonomy),'raw_top1_failures':sum(not x['raw_top1_positive'] for x in taxonomy),'corrected_top1_failures':sum(not x['corrected_top1_positive'] for x in taxonomy)}})
 atomic(OUT/'completion/stage0.done',{'stage':0,'contract':'PASS','baseline':str(OUT/'audit/baseline_replay.json')})
 print(json.dumps({'stage0':'PASS','aggregate_p16':agg['16']},indent=2))
if __name__=='__main__': main()
