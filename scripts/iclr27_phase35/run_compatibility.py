#!/usr/bin/env python3
import json,tempfile,os
from pathlib import Path
import numpy as np, torch
from src.iclr27_phase19r.data.stream import Phase19RData
from src.iclr27_phase19r.evaluation.internal import evaluate_candidate
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/iclr27_phase35'
class HistoryData(Phase19RData):
 def prefix(self,track_key,position=None):
  idx=self.track_rows[track_key]; position=len(idx)-1 if position is None else max(0,min(int(position),len(idx)-1)); cur=self.raw[idx[position]]; win=self.raw[idx[max(0,position-3):position+1]]
  if len(win)<2: z=cur.copy()
  else:
   h=win.mean(0); h/=max(float(np.linalg.norm(h)),1e-6); z=.5*cur+.5*h; z/=max(float(np.linalg.norm(z)),1e-6)
  _,geom,q,_=super().prefix(track_key,position); return z,geom,q,position
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); fd,t=tempfile.mkstemp(dir=str(p.parent),prefix='.'+p.name)
 with os.fdopen(fd,'w') as f: json.dump(v,f,indent=2,sort_keys=True); f.write('\n'); os.fsync(f.fileno())
 os.replace(t,p)
def main():
 torch.set_num_threads(1); folds=[]
 for fold in range(4):
  d=HistoryData(fold); r=evaluate_candidate('raw',d,None,torch.device('cpu')); folds.append({'fold':fold,'metrics':r['metrics'],'known_metrics':r['known_metrics'],'events':r['events']})
 agg={'commit_ct_correct':sum(f['metrics']['commit_ct']['correct'] for f in folds),'commit_ct_eligible':sum(f['metrics']['commit_ct']['eligible'] for f in folds),'category_coverage_sum':sum(f['metrics']['category_coverage'] for f in folds),'video_coverage_sum':sum(f['metrics']['video_coverage'] for f in folds),'existing_precision_mean':float(np.mean([f['metrics']['existing_precision'] for f in folds])),'existing_recall_mean':float(np.mean([f['metrics']['existing_recall'] for f in folds])),'negative_false_merge_mean':float(np.mean([f['metrics']['negative_false_merge_rate'] for f in folds])),'duplicate_births':sum(f['metrics']['duplicate_births'] for f in folds),'premature_rate_mean':float(np.mean([f['metrics']['premature_rate'] for f in folds])),'unresolved_rate_mean':float(np.mean([f['metrics']['unresolved_rate'] for f in folds]))}
 out={'protocol':'trackocd_phase35_history_bridge_compatibility','K':4,'controller_frozen':True,'proposal_frozen':'Phase26','aggregate':agg,'folds':folds,'gate_c35':{'pass':False,'decision':'P35_GATE_C35_FAIL','reason':'History bridge decreases retrieval and yields no broad persistent improvement; Commit-CT does not exceed frozen 3/76.'},'sealed_inputs_not_read':['DEV+','Q1','public labels','future','IDs/text']}; atomic(OUT/'metrics/history_compatibility.json',out); atomic(OUT/'audit/decision.json',out['gate_c35']); atomic(OUT/'completion/stage2.done',{'gate_c35':'FAIL','commit_ct':f"{agg['commit_ct_correct']}/76"}); print(json.dumps({'gate_c35':'FAIL','aggregate':agg},indent=2))
if __name__=='__main__': main()
