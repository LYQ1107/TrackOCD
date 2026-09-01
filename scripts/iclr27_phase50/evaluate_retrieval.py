#!/usr/bin/env python3
"""Frozen TRAIN-disjoint retrieval replay for Phase50 checkpoints.

This is a diagnostic Gate-R screen, not the 76-event causal evaluator.  Query
support lists come from the fixed validation manifest and no validation labels
are passed to the model; labels are used only to score the retrieval result.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
import numpy as np
import torch
from scripts.iclr27_phase30.run_stage1_diagnostics import load_tracks, track_metadata
from src.iclr27_phase50.end_to_end import EndToEndTrackOCD

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/iclr27_phase50"
PREFIXES = (1, 2, 4, 8, 16)

def atomic(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=path.parent)
    with os.fdopen(fd,'w') as f:
        json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def vec(key, meta, feats, prefix):
    inds = np.asarray(meta[key]['rows'][:min(prefix,16)], dtype=np.int64)
    z = feats[inds].mean(0) if len(inds) else np.zeros(feats.shape[1], np.float32)
    return (z/max(float(np.linalg.norm(z)),1e-8)).astype(np.float32)

def seq(key, meta, feats, prefix):
    inds=np.asarray(meta[key]['rows'][:min(prefix,16)],dtype=np.int64)
    return feats[inds].astype(np.float32) if len(inds) else np.zeros((1,feats.shape[1]),np.float32)

@torch.no_grad()
def main():
    ap = __import__('argparse').ArgumentParser(); ap.add_argument('--tag',default='e2e_formal'); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--checkpoint-pattern',default='e2e_formal_f{fold}_best.pt'); args=ap.parse_args()
    torch.set_num_threads(1)
    rows, tracks, feats = load_tracks(); meta=track_metadata(rows,tracks)
    dev=torch.device(args.device if torch.cuda.is_available() else 'cpu'); folds=[]
    for fold in range(4):
        man=json.loads((ROOT/f'outputs/iclr27_phase30/manifests/episode_manifest_f{fold}.json').read_text())
        val=[r for r in man['records'] if r.get('split')=='val' and r.get('kind')=='multi_positive_cross_video' and r.get('query_track_key') in meta]
        keys=sorted({r['query_track_key'] for r in val}); supports={r['query_track_key']:[k for k in r.get('support_track_keys',[]) if k in meta] for r in val}
        ckpath=OUT/'checkpoints'/args.checkpoint_pattern.format(fold=fold)
        if not ckpath.exists(): raise FileNotFoundError(ckpath)
        model=EndToEndTrackOCD().to(dev); model.load_state_dict(torch.load(ckpath,map_location='cpu',weights_only=False)['model']); model.eval(); vids=np.asarray([meta[k]['video'] for k in keys]); cats=np.asarray([meta[k]['category'] for k in keys]); per={}
        for p in PREFIXES:
            raw=np.asarray([vec(k,meta,feats,p) for k in keys],np.float32)
            learned=[]; valid_support=[]
            for i,k in enumerate(keys):
                ss=[vec(s,meta,feats,p) for s in supports.get(k,[]) if s in meta]
                if ss:
                    q=torch.from_numpy(seq(k,meta,feats,p)).unsqueeze(0).to(dev); s=torch.from_numpy(np.asarray(ss,np.float32)).unsqueeze(0).to(dev); m=torch.ones((1,s.shape[1]),dtype=torch.bool,device=dev)
                    learned.append(model(q,s,m)['semantic'].cpu().numpy()[0]); valid_support.append(True)
                else:
                    learned.append(raw[i]); valid_support.append(False)
            learned=np.asarray(learned,np.float32); raw_sim=raw@raw.T; learned_sim=learned@learned.T
            raw_r1=[]; raw_r5=[]; raw_ap=[]; lr1=[]; lr5=[]; lap=[]; raw_gap=[]; learned_gap=[]; unsafe=[]; bridge_delta=[]; covered=0
            for i,k in enumerate(keys):
                cand=np.where((np.arange(len(keys))!=i)&(vids!=vids[i]))[0]; pos=cand[cats[cand]==cats[i]]; neg=cand[cats[cand]!=cats[i]]
                if len(pos)==0 or len(neg)==0: continue
                covered+=1; ro=cand[np.argsort(raw_sim[i,cand])[::-1]]; lo=cand[np.argsort(learned_sim[i,cand])[::-1]]
                rh=np.isin(ro,pos).astype(float); lh=np.isin(lo,pos).astype(float)
                raw_r1.append(float(rh[0])); raw_r5.append(float(rh[:5].max(initial=0))); lr1.append(float(lh[0])); lr5.append(float(lh[:5].max(initial=0)))
                rc=np.cumsum(rh); lc=np.cumsum(lh); raw_ap.append(float(np.sum(rc/(np.arange(len(rh))+1)*rh)/max(len(pos),1))); lap.append(float(np.sum(lc/(np.arange(len(lh))+1)*lh)/max(len(pos),1)))
                raw_gap.append(float(raw_sim[i,pos].max()-raw_sim[i,neg].max())); learned_gap.append(float(learned_sim[i,pos].max()-learned_sim[i,neg].max())); unsafe.append(float(ro[0] in set(pos.tolist()) and lo[0] in set(neg.tolist())))
                bridge_delta.append(float(learned_sim[i,pos].max()-raw_sim[i,pos].max()))
            per[str(p)]={'queries':len(raw_r1),'support_available':int(sum(valid_support)),'support_rate':float(np.mean(valid_support) if valid_support else 0.0),'raw':{'r1':float(np.mean(raw_r1) if raw_r1 else 0),'r5':float(np.mean(raw_r5) if raw_r5 else 0),'map':float(np.mean(raw_ap) if raw_ap else 0),'hard_gap':float(np.mean(raw_gap) if raw_gap else 0)},'learned':{'r1':float(np.mean(lr1) if lr1 else 0),'r5':float(np.mean(lr5) if lr5 else 0),'map':float(np.mean(lap) if lap else 0),'hard_gap':float(np.mean(learned_gap) if learned_gap else 0),'unsafe_flip_rate':float(np.mean(unsafe) if unsafe else 0),'mean_positive_delta':float(np.mean(bridge_delta) if bridge_delta else 0)}}
        folds.append({'fold':fold,'validation_queries':len(keys),'prefix':per,'checkpoint':str(ckpath)})
    aggregate={}
    for p in PREFIXES:
        fs=[f['prefix'][str(p)] for f in folds]
        aggregate[str(p)]={'raw':{m:float(np.mean([x['raw'][m] for x in fs])) for m in ('r1','r5','map','hard_gap')},'learned':{m:float(np.mean([x['learned'][m] for x in fs])) for m in ('r1','r5','map','hard_gap','unsafe_flip_rate','mean_positive_delta')},'folds':fs}
    p16=aggregate['16']; raw16=p16['raw']; learned16=p16['learned']
    same_direction=sum(float(f['prefix']['16']['learned']['r1'])>=float(f['prefix']['16']['raw']['r1']) and float(f['prefix']['16']['learned']['map'])>=float(f['prefix']['16']['raw']['map']) for f in folds)
    decision = {'r1_margin':float(learned16['r1']-raw16['r1']),'map_margin':float(learned16['map']-raw16['map']),'same_direction_folds':int(same_direction),'hard_gap_non_worse':bool(all(f['prefix']['16']['learned']['hard_gap']>=f['prefix']['16']['raw']['hard_gap']-1e-12 for f in folds)),'unsafe_flip_zero':bool(learned16['unsafe_flip_rate']==0.0)}
    # Avoid using held-event outcomes for selection: this Gate R decision uses
    # only the fixed validation manifests above.
    decision['gate_r50']='PASS' if decision['r1_margin']>=0.02 and decision['map_margin']>=0.01 and same_direction>=3 and decision['hard_gap_non_worse'] and decision['unsafe_flip_zero'] else 'FAIL'
    result={'phase':50,'protocol':'phase50_train_disjoint_retrieval','prefixes':list(PREFIXES),'folds':folds,'aggregate':aggregate,'decision':decision,'sealed_inputs_not_read':['DEV+','Q1','public new-model labels','held event outcomes','future rows/tracks','category/text/ID inputs']}
    atomic(OUT/'metrics/phase50_retrieval.json',result); atomic(OUT/'completion/retrieval.done',{'phase':50,'gate_r50':decision['gate_r50'],'r1_margin':decision['r1_margin'],'map_margin':decision['map_margin']}); print(json.dumps({'gate_r50':decision['gate_r50'],'p16':p16},indent=2))

if __name__=='__main__': main()
