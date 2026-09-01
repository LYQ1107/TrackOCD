#!/usr/bin/env python3
"""Phase46 C2: frozen Phase46 gate vectors with unchanged Phase19R controller."""
import argparse, json, os, tempfile
from pathlib import Path
import numpy as np, torch
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate, load_events
from src.iclr27_phase46.selective import ConditionalLogitGate
from src.iclr27_phase41.bridge import SafetyVectorBridge
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase46'
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.replace(t,p)
class GatedData(Phase19RData):
 def __init__(self,fold,gate,bridge):
  super().__init__(fold=fold,final=False); self.gate=gate.eval(); self.bridge=bridge.eval(); self.support_by_video={}
  man=json.load(open(ROOT/'outputs/iclr27_phase38/audit/support_stream_manifest.json'))
  for r in man['policies']['PRIOR_COMPLETED_TRACK']:
   self.support_by_video.setdefault(int(r['target_video']),r['support_track_keys'])
  self._support_cache={}
 def _support_context(self,video,prefix):
  key=(int(video),int(prefix))
  if key in self._support_cache:return self._support_cache[key]
  ks=[k for k in self.support_by_video.get(int(video),[]) if k in self.track_rows]
  vs=[]
  for k in ks:
   raw,_,_,_=super().prefix(k,min(prefix,len(self.track_rows[k])-1)); vs.append(raw)
  if not vs:return None,None
  arr=np.asarray(vs); ctx=arr.mean(0); ctx/=max(float(np.linalg.norm(ctx)),1e-8); self._support_cache[key]=(ctx,arr); return ctx,arr
 def prefix(self,track_key,position=None):
  raw,geom,q,pos=super().prefix(track_key,position); row=self.rows[self.track_rows[track_key][pos]]; ctx,sv=self._support_context(int(row['video_id']),min(pos+1,16))
  if ctx is None or len(sv)<2:return raw,geom,q,pos
  raw_score=sv@raw; sq=float(raw_score.max()); ctx_t=torch.tensor(ctx).view(1,-1); raw_t=torch.tensor(raw).view(1,-1); sv_t=torch.tensor(sv)
  with torch.no_grad():
   z,a,_=self.bridge(raw_t,ctx_t,torch.tensor([sq]),torch.tensor([sq]),True); bv=z.numpy()[0]
   rs=np.sort(raw_score)[::-1]; bs=np.sort(sv@bv)[::-1]; rm=float(rs[0]-rs[1]); bm=float(bs[0]-bs[1]); logit=self.gate(torch.tensor([rm]),torch.tensor([bm]),torch.tensor([sq]),a,torch.zeros(1)); use=bool(logit.item()>=0.)
  return (bv if use else raw),geom,q,pos
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); ap.add_argument('--tag',default='phase46_c2_v1'); a=ap.parse_args(); fold=a.fold; dev=torch.device('cpu')
 gate=ConditionalLogitGate(); gate.load_state_dict(torch.load(OUT/'checkpoints'/f'phase46_formal_v1_f{fold}_best.pt',map_location='cpu',weights_only=False)['model']); bridge=SafetyVectorBridge(); bridge.load_state_dict(torch.load(ROOT/f'outputs/iclr27_phase41/checkpoints/gate_formal_fix2_f{fold}_best.pt',map_location='cpu',weights_only=False)['model'])
 gated=GatedData(fold,gate,bridge); raw=Phase19RData(fold=fold,final=False); ck=ROOT/'outputs/iclr27_phase19r/checkpoints'/f'fold{fold}_best_internal.pt'; main=evaluate_candidate('main',gated,ck,dev); base=evaluate_candidate('raw',raw,None,dev)
 out={'phase':46,'fold':fold,'protocol':'phase46_c2_unchanged_controller_frozen_gate','main':{'metrics':main['metrics'],'known_metrics':main['known_metrics'],'events':len(main['records']),'checkpoint':str(ck)},'raw':{'metrics':base['metrics'],'events':len(base['records'])},'positive_events':sum(r['kind']=='positive_existing' for r in main['records']),'negative_events':sum(r['kind']=='negative_new' for r in main['records']),'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text/held GT input']}
 atomic(OUT/'metrics'/f'controller_f{fold}.json',out); atomic(OUT/'completion'/f'controller_f{fold}.done',{'fold':fold,'positive_events':out['positive_events'],'commit_ct':out['main']['metrics']['commit_ct']}); print(json.dumps({'fold':fold,'main_commit_ct':out['main']['metrics']['commit_ct'],'raw_commit_ct':out['raw']['metrics']['commit_ct']},sort_keys=True))
if __name__=='__main__': main()
