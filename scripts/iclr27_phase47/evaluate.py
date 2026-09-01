#!/usr/bin/env python3
import json, tempfile, os
from pathlib import Path
import numpy as np, torch
from src.iclr27_phase19r.data.stream import Phase19RData
from scripts.iclr27_phase46.evaluate_controller import GatedData
from src.iclr27_phase46.selective import ConditionalLogitGate
from src.iclr27_phase41.bridge import SafetyVectorBridge
from src.iclr27_phase47.correspondence import DomainAlignedEncoder
from scripts.iclr27_phase47.train_correspondence import build_sets, retrieval
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase47'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 rows=[]
 for fold in range(4):
  gate=ConditionalLogitGate(); gate.load_state_dict(torch.load(ROOT/'outputs/iclr27_phase46/checkpoints'/f'phase46_formal_v1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); gd=GatedData(fold,gate,bridge); fit,val=build_sets(gd,fold); model=DomainAlignedEncoder(); model.load_state_dict(torch.load(OUT/f'checkpoints/phase47_formal_v1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); model.eval(); keys=[k for c in val for k in val[c]]; gated_vec={k:gd.prefix(k,min(15,len(gd.track_rows[k])-1))[0] for k in keys}; rawd=Phase19RData(fold); raw_vec={k:rawd.prefix(k,min(15,len(rawd.track_rows[k])-1))[0] for k in keys}; learned=retrieval(model,gated_vec,val,torch.device('cpu')); base_model=DomainAlignedEncoder(); base_model.net[-1].weight.data.zero_(); base_model.net[-1].bias.data.zero_(); # not used; compute cosine directly
  cats={k:c for c,ks in val.items() for k in ks}; vids={k:int(k.split(':')[0][1:]) for k in keys}; arr=np.asarray([raw_vec[k] for k in keys]); s=arr@arr.T; r1=[];aps=[];gaps=[]
  for i,k in enumerate(keys):
   cand=[j for j,z in enumerate(keys) if j!=i and vids[z]!=vids[k]]; pos=[j for j in cand if cats[keys[j]]==cats[k]]; neg=[j for j in cand if cats[keys[j]]!=cats[k]]
   if not pos or not neg: continue
   order=np.asarray(cand)[np.argsort(s[i,cand])[::-1]]; hit=np.asarray([int(j in pos) for j in order]); r1.append(float(hit[0])); c=np.cumsum(hit); aps.append(float(np.sum(c/(np.arange(len(hit))+1)*hit)/len(pos))); gaps.append(float(s[i,pos].max()-s[i,neg].max()))
  raw={'r1':float(np.mean(r1)),'map':float(np.mean(aps)),'hard_gap':float(np.mean(gaps)),'queries':len(r1)}; rows.append({'fold':fold,'raw':raw,'learned':learned,'fit_tracks':sum(len(x) for x in fit.values()),'validation_tracks':len(keys)})
 agg={n:{m:float(np.mean([r[n][m] for r in rows])) for m in ('r1','map','hard_gap')} for n in ('raw','learned')}; atomic(OUT/'metrics/phase47_retrieval.json',{'protocol':'phase47_domain_aligned_correspondence','folds':rows,'aggregate':agg,'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT']}); atomic(OUT/'completion/retrieval.done',{'aggregate':agg}); print(json.dumps({'aggregate':agg,'folds':rows},indent=2))
if __name__=='__main__': main()
