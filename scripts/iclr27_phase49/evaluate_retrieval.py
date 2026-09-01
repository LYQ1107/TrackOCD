#!/usr/bin/env python3
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
import numpy as np, torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase49.residual import RawPreservingResidualBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase49'; PREFIXES=(1,2,4,8,16); TAG=os.environ.get('PHASE49_TAG','phase49_formal')
def atomic(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=p.parent,prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def vec(k,m,f,p):
 z=f[np.asarray(m[k]['rows'][:min(p,16)])].mean(0); return z/max(float(np.linalg.norm(z)),1e-8)
@torch.no_grad()
def main():
 torch.set_num_threads(1); rows,tr,feats=load_tracks(); meta=track_metadata(rows,tr); dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); folds=[]
 for fold in range(4):
  man=json.loads((ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json').read_text()); val=[r for r in man['records'] if r.get('split')=='val' and r.get('kind')=='multi_positive_cross_video']; keys=sorted({r['query_track_key'] for r in val if r['query_track_key'] in meta}); supp={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val if r.get('query_track_key') in meta}; ck=torch.load(OUT/f'checkpoints/{TAG}_f{fold}_best.pt',map_location='cpu',weights_only=False); model=RawPreservingResidualBridge().to(dev); model.load_state_dict(ck['model']); model.eval(); vids=np.array([meta[k]['video'] for k in keys]); cats=np.array([meta[k]['category'] for k in keys]); pout={}
  for p in PREFIXES:
   base=np.asarray([vec(k,meta,feats,p) for k in keys],np.float32); sim=base@base.T; raw=[]; learned=[]; gaps=[]; resid=[]; alphas=[]; unsafe=0; changed=0
   for i,k in enumerate(keys):
    cand=np.where((np.arange(len(keys))!=i)&(vids!=vids[i]))[0]; pos=cand[cats[cand]==cats[i]]; neg=cand[cats[cand]!=cats[i]]
    if len(pos)==0 or len(neg)==0: continue
    ss=[vec(x,meta,feats,p) for x in supp.get(k,[]) if x in meta]; ss=ss or [base[i]]; q=torch.tensor(base[i],device=dev).view(1,-1); st=torch.tensor(np.asarray(ss),device=dev).unsqueeze(0); z,a,r=model(q,st,torch.ones(1,len(ss),device=dev,dtype=torch.bool),True); scores=(z@torch.tensor(base[cand],device=dev).T).cpu().numpy()[0]; rs=sim[i,cand]; order=np.argsort(scores)[::-1]; ro=np.argsort(rs)[::-1]; hit=np.isin(cand[order],pos).astype(float); rh=np.isin(cand[ro],pos).astype(float); unsafe += int(rh[0] > hit[0]); changed += int(rh[0] != hit[0]); learned.append((hit[0],hit[:5].max(initial=0),np.sum(np.cumsum(hit)/(np.arange(len(hit))+1)*hit)/max(len(pos),1))); raw.append((rh[0],rh[:5].max(initial=0),np.sum(np.cumsum(rh)/(np.arange(len(rh))+1)*rh)/max(len(pos),1))); gaps.append(float(scores[np.isin(cand,pos)].max()-scores[np.isin(cand,neg)].max())); resid.append(float(r.abs().mean())); alphas.append(float(a.mean()))
   pout[str(p)]={'queries':len(learned),'raw':{'r1':float(np.mean([x[0] for x in raw])) if raw else 0,'r5':float(np.mean([x[1] for x in raw])) if raw else 0,'map':float(np.mean([x[2] for x in raw])) if raw else 0},'learned':{'r1':float(np.mean([x[0] for x in learned])) if learned else 0,'r5':float(np.mean([x[1] for x in learned])) if learned else 0,'map':float(np.mean([x[2] for x in learned])) if learned else 0,'hard_gap':float(np.mean(gaps)) if gaps else 0,'residual_abs':float(np.mean(resid)) if resid else 0,'alpha':float(np.mean(alphas)) if alphas else 0,'unsafe_flip_rate':float(unsafe/max(len(learned),1)),'top1_change_rate':float(changed/max(len(learned),1))}}
  folds.append({'fold':fold,'validation_tracklets':len(keys),'prefix':pout})
 agg={}
 for p in PREFIXES:
  fs=[f['prefix'][str(p)] for f in folds]; agg[str(p)]={'raw':{m:float(np.mean([x['raw'][m] for x in fs])) for m in ('r1','r5','map')},'learned':{m:float(np.mean([x['learned'][m] for x in fs])) for m in ('r1','r5','map')},'hard_gap':float(np.mean([x['learned']['hard_gap'] for x in fs])),'residual_abs':float(np.mean([x['learned']['residual_abs'] for x in fs])),'alpha':float(np.mean([x['learned']['alpha'] for x in fs]))}
 x={'phase':49,'protocol':'phase49_raw_preserving_residual','checkpoint_tag':TAG,'prefixes':list(PREFIXES),'folds':folds,'aggregate':agg,'gate_r49':'FAIL','controller_run':False,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}; atomic(OUT/f'metrics/phase49_retrieval_{TAG}.json',x); atomic(OUT/'metrics/phase49_retrieval.json',x); atomic(OUT/'metrics/phase49_controller.json',{'phase':49,'status':'NOT_RUN_GATE_R49_FAIL','reason':'retrieval gate failed; controller remains frozen','sealed_inputs_not_read':x['sealed_inputs_not_read']}); atomic(OUT/'completion/retrieval.done',{'phase':49,'gate':'FAIL','tag':TAG}); print(json.dumps(agg['16'],indent=2))
if __name__=='__main__': main()
